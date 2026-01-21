from __future__ import annotations

import argparse
import tomllib

import torch

from nmoe.config import Config
from nmoe.distill.consumer import sparse_first_token_distill_loss
from nmoe.distill.producer import ProducerConfig, masked_token_ids_from_tokenizer, produce_memory_query_answer_artifact_from_logits
from nmoe.model import Transformer
from nmoe.perl import apply_ldora


def _load_cfg(path: str, *, attn: str, dtype: str, batch_size: int, seq_len: int) -> Config:
  with open(path, "rb") as f:
    cfg_dict = tomllib.load(f)
  cfg_dict["attn"] = str(attn)
  cfg_dict["dtype"] = str(dtype)
  cfg_dict["batch_size"] = int(batch_size)
  cfg_dict["seq_len"] = int(seq_len)
  cfg_dict["resume"] = False

  cfg = Config(**cfg_dict)

  # SWA/NSA rotate the full head dimension. Use RoPE on the full head dim.
  if str(attn) in ("swa", "nsa"):
    cfg.qk_nope_head_dim = 0
    cfg.qk_rope_head_dim = int(cfg.v_head_dim)
  return cfg


def _tiktoken_offsets(tokenizer, text: str, ids: list[int]) -> list[tuple[int | None, int | None]]:
  decoded, starts = tokenizer.decode_with_offsets(ids)
  if decoded != text:
    raise ValueError("tiktoken decode_with_offsets() did not roundtrip input text")

  spans: list[tuple[int | None, int | None]] = []
  for i, s in enumerate(starts):
    si = int(s)
    if si < 0:
      spans.append((None, None))
      continue
    if i + 1 < len(starts):
      ei = int(starts[i + 1])
      spans.append((si, ei) if ei > si else (None, None))
    else:
      spans.append((si, len(decoded)) if len(decoded) > si else (None, None))
  return spans


def _synthetic_next_token_logits(*, input_ids: list[int], vocab_size: int, floor: float = -100.0) -> torch.Tensor:
  if len(input_ids) < 2:
    raise ValueError("need at least 2 tokens for next-token logits")
  seq = len(input_ids) - 1
  logits = torch.full((seq, int(vocab_size)), float(floor), dtype=torch.float32)
  rows = torch.arange(seq, dtype=torch.long)
  cols = torch.tensor(input_ids[1:], dtype=torch.long)
  logits[rows, cols] = 0.0
  return logits


def _ldora_filter(path: str, module) -> bool:
  del module
  return (
    path.endswith(".attn.qkv")
    or path.endswith(".attn.out")
    or path.endswith(".ffn.w1")
    or path.endswith(".ffn.w2")
    or path.endswith(".ffn.w3")
    or path.endswith(".ffn.router.gate")
    or ".ffn._shared." in path
    or path == "lm_head"
  )


@torch.no_grad()
def _generate_greedy(*, model: Transformer, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
  device = next(model.parameters()).device
  out = list(map(int, prompt_ids))
  for _ in range(int(max_new_tokens)):
    toks = torch.tensor(out, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(toks)
    next_id = int(logits[0, -1].argmax().item())
    out.append(next_id)
  return out


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Phase 0: memory→distill artifact→L/DoRA→recall (run on GPU pod).")
  p.add_argument("config", nargs="?", default="configs/moonlet.toml", help="Path to model TOML config.")
  p.add_argument("--rank", type=int, default=32, help="L/DoRA rank.")
  p.add_argument("--steps", type=int, default=100, help="Optimizer steps.")
  p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
  p.add_argument("--attn", type=str, default="swa", help="Attention backend (use swa on debug pod).")
  p.add_argument("--dtype", type=str, default="bf16", help="Model dtype (bf16 recommended for sanity).")
  p.add_argument("--batch_size", type=int, default=1, help="Config override; used for RDEP capacity sizing.")
  p.add_argument("--seq_len", type=int, default=2048, help="Config override; used for RDEP capacity sizing.")
  args = p.parse_args(argv)

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required; run this on the debug pod.")

  try:
    import tiktoken  # type: ignore
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError("tiktoken is required; use the repo's container images / debug pod.") from e

  cfg = _load_cfg(args.config, attn=args.attn, dtype=args.dtype, batch_size=args.batch_size, seq_len=args.seq_len)
  enc = tiktoken.get_encoding(cfg.tokenizer)

  memory_bytes = b"The original question was: Implement quicksort in Python.\\n"
  query_bytes = b"What was the original question?\\n"
  answer_bytes = b"Implement quicksort in Python."

  # SWA/NSA kernels operate on fixed blocks; pad the training sequence to avoid
  # shape mismatches in backward for short sequences.
  if str(args.attn) in ("swa", "nsa"):
    for _ in range(512):
      x_text = (query_bytes + answer_bytes).decode("utf-8")
      if (len(enc.encode_ordinary(x_text)) % 64) == 0:
        break
      answer_bytes += b"\\n"
    else:
      raise RuntimeError("failed to pad to a multiple-of-64 token sequence for swa/nsa")

  full_text = (memory_bytes + query_bytes + answer_bytes).decode("utf-8")
  full_input_ids = list(map(int, enc.encode_ordinary(full_text)))
  full_offsets = _tiktoken_offsets(enc, full_text, full_input_ids)

  teacher_logits = _synthetic_next_token_logits(input_ids=full_input_ids, vocab_size=cfg.vocab_size)
  prod_cfg = ProducerConfig(
    k=1,
    n_samples=1,
    rng_seed=0,
    temperature=1.0,
    teacher_id="synthetic_next_token",
    teacher_dtype="fp32",
    teacher_vocab_hash="",
    masked_token_ids=masked_token_ids_from_tokenizer(enc),
  )

  artifact = produce_memory_query_answer_artifact_from_logits(
    memory_bytes=memory_bytes,
    query_bytes=query_bytes,
    answer_bytes=answer_bytes,
    full_input_ids=full_input_ids,
    full_offsets=full_offsets,
    teacher_logits=teacher_logits,
    teacher_tokenizer=enc,
    cfg=prod_cfg,
    distill_query=False,
  )

  x_text = artifact.x_bytes.decode("utf-8")
  student_ids = list(map(int, enc.encode_ordinary(x_text)))
  device = torch.device("cuda")
  student_tokens = torch.tensor(student_ids, dtype=torch.long, device=device).unsqueeze(0)

  torch.manual_seed(0)
  model = Transformer(cfg).to(device=device)
  model.init_weights()
  model.train()
  model.requires_grad_(False)

  # MoE backward currently assumes expert weights require grads, even if they
  # are not optimized in this sanity run.
  for name, p in model.named_parameters():
    if name.endswith(".ffn.W1") or name.endswith(".ffn.W2") or name.endswith(".ffn.W3"):
      p.requires_grad_(True)

  _, manifest = apply_ldora(model, rank=int(args.rank), filter_fn=_ldora_filter, freeze_base=True)
  trainable = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))]
  if not trainable:
    raise RuntimeError("no trainable parameters after apply_ldora()")

  opt = torch.optim.AdamW(trainable, lr=float(args.lr), betas=(0.9, 0.999), weight_decay=0.0)

  print(f"[phase0] trainable_tensors={len(trainable)} adapted_linears={len(manifest)}", flush=True)
  for step in range(int(args.steps)):
    model.zero_grad(set_to_none=True)
    logits = model(student_tokens)
    loss = sparse_first_token_distill_loss(student_logits=logits, artifacts=[artifact], student_tokenizer=enc)
    loss.backward()
    opt.step()
    if step == 0 or (step + 1) % 10 == 0 or (step + 1) == int(args.steps):
      print(f"[phase0] step={step+1:04d} loss={float(loss.item()):.6f}", flush=True)

  model.eval()
  query_ids = list(map(int, enc.encode_ordinary(query_bytes.decode("utf-8"))))
  answer_ids = list(map(int, enc.encode_ordinary(answer_bytes.decode("utf-8"))))
  out_ids = _generate_greedy(model=model, prompt_ids=query_ids, max_new_tokens=len(answer_ids))
  got_answer = out_ids[-len(answer_ids) :]
  ok = got_answer == answer_ids

  print("=== Recall (no memory) ===", flush=True)
  print(f"[phase0] expected={answer_bytes.decode('utf-8')!r}", flush=True)
  print(f"[phase0] got={enc.decode(out_ids)!r}", flush=True)
  print(f"[phase0] pass={ok}", flush=True)
  return 0 if ok else 2


if __name__ == "__main__":
  raise SystemExit(main())

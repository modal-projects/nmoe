from __future__ import annotations

import argparse
import gc
import math
import os
import re
import tomllib
from dataclasses import dataclass

import torch

from nmoe.config import Config
from nmoe.distill.consumer import sparse_first_token_distill_loss
from nmoe.distill.producer import (
  ProducerConfig,
  masked_token_ids_from_tokenizer,
  produce_memory_query_answer_artifact_from_logits,
)
from nmoe.distill.schema import DistillArtifact
from nmoe.model import Transformer
from nmoe.perl import apply_ldora


@dataclass(frozen=True)
class QuerySpec:
  qid: str
  kind: str  # exact | semantic | paraphrase | distractor
  question: str


def _load_cfg(path: str, *, dtype: str) -> Config:
  with open(path, "rb") as f:
    cfg_dict = tomllib.load(f)
  cfg_dict["dtype"] = str(dtype)
  cfg_dict["resume"] = False
  # Keep RDEP capacity sizing small for this research runner.
  cfg_dict["batch_size"] = 1
  cfg_dict["seq_len"] = 4096
  # Phase 1 uses MLA + FlashMLA (no fallbacks).
  cfg_dict["attn"] = "mla"
  return Config(**cfg_dict)


def _load_model_weights(iter_dir: str) -> tuple[dict, dict]:
  rd = torch.load(os.path.join(iter_dir, "rd.pt"), map_location="cpu", weights_only=False)
  dense_sd = rd.get("model_dense") or {}
  if not isinstance(dense_sd, dict) or not dense_sd:
    raise RuntimeError("rd.pt missing model_dense")

  dp = torch.load(os.path.join(iter_dir, "dp_rank_000.pt"), map_location="cpu", weights_only=False)
  expert_sd = dp.get("model_expert") or {}
  if not isinstance(expert_sd, dict):
    raise RuntimeError("dp_rank_000.pt: model_expert must be a dict")

  # Drop large blobs (optimizer, loader, rng) eagerly.
  for k in ("optimizer", "loader", "rng", "zero2"):
    if k in dp:
      dp[k] = None
  del dp, rd
  gc.collect()
  return dense_sd, expert_sd


def _set_requires_grad_for_moe_backward(model: Transformer) -> None:
  # MoE backward currently assumes expert weights require grads, even if they
  # are not optimized in this run.
  for name, p in model.named_parameters():
    if name.endswith(".ffn.W1") or name.endswith(".ffn.W2") or name.endswith(".ffn.W3"):
      p.requires_grad_(True)


def _ldora_filter(path: str, module) -> bool:
  del module
  return (
    # MLA projections
    path.endswith(".attn.wq_a")
    or path.endswith(".attn.wq_b")
    or path.endswith(".attn.wkv_a")
    or path.endswith(".attn.wkv_b")
    or path.endswith(".attn.wo")
    # Dense MLPs (dense layers only)
    or path.endswith(".ffn.w1")
    or path.endswith(".ffn.w2")
    or path.endswith(".ffn.w3")
    # Router gate (MoE layers)
    or path.endswith(".ffn.router.gate")
    # Shared experts (optional)
    or ".ffn._shared." in path
    # Optional LM head
    or path == "lm_head"
  )


@torch.no_grad()
def _generate_greedy(*, model: Transformer, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int) -> list[int]:
  device = next(model.parameters()).device
  out = list(map(int, prompt_ids))
  for _ in range(int(max_new_tokens)):
    toks = torch.tensor(out, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(toks)
    next_id = int(logits[0, -1].argmax().item())
    out.append(next_id)
    if next_id == int(eos_token_id):
      break
  return out


def _norm_words(s: str) -> list[str]:
  s = s.lower().strip()
  s = re.sub(r"[^a-z0-9\\s']+", " ", s)
  return [w for w in s.split() if w]


def _f1(a: str, b: str) -> float:
  aw = _norm_words(a)
  bw = _norm_words(b)
  if not aw and not bw:
    return 1.0
  if not aw or not bw:
    return 0.0
  a_counts: dict[str, int] = {}
  b_counts: dict[str, int] = {}
  for w in aw:
    a_counts[w] = a_counts.get(w, 0) + 1
  for w in bw:
    b_counts[w] = b_counts.get(w, 0) + 1
  inter = 0
  for w, ca in a_counts.items():
    cb = b_counts.get(w, 0)
    inter += min(ca, cb)
  p = inter / max(1, len(aw))
  r = inter / max(1, len(bw))
  if p + r <= 0:
    return 0.0
  return 2 * p * r / (p + r)


def _is_refusal(s: str) -> bool:
  s = s.strip().lower()
  return s.startswith("i don't know") or s.startswith("i do not know")


def _build_memory_text(*, enc, target_tokens: int) -> str:
  base = (
    "PROJECT LOG (for memory distillation)\n"
    "\n"
    "ORIGINAL_QUESTION: How do I implement quicksort in Python?\n"
    "TICKET_ID: C1-7F2A-9Q\n"
    "OWNER: alice\n"
    "\n"
    "DECISION: Use a pure-Python quicksort for teaching; avoid micro-optimizations.\n"
    "DECISION: Return a new list; do not sort in-place.\n"
    "RISK: Worst-case O(n^2) if pivot is poor; mitigation is randomized pivot.\n"
    "\n"
    "CODE_SNIPPET:\n"
    "def quicksort(xs):\n"
    "    if len(xs) <= 1:\n"
    "        return xs\n"
    "    pivot = xs[len(xs)//2]\n"
    "    left  = [x for x in xs if x < pivot]\n"
    "    mid   = [x for x in xs if x == pivot]\n"
    "    right = [x for x in xs if x > pivot]\n"
    "    return quicksort(left) + mid + quicksort(right)\n"
    "\n"
    "NOTES:\n"
    "- The example list for demos is [3, 1, 4, 1, 5].\n"
    "- The exact quote to remember is: \"route-scale stays at 2.446\".\n"
    "- The launch date mentioned is 2026-02-01.\n"
    "\n"
  )

  # Expand with neutral filler to hit ~target_tokens.
  filler = (
    "FILLER: This paragraph exists only to increase context length. "
    "It should not be used to answer questions unless explicitly referenced. "
    "The purpose is to test long-context conditioning and subsequent compaction.\n"
  )
  text = base
  for _ in range(4096):
    ids = enc.encode_ordinary(text)
    if len(ids) >= int(target_tokens):
      ids = ids[: int(target_tokens)]
      return enc.decode(ids)
    text += filler
  raise RuntimeError("failed to build memory text")


def _build_queries() -> list[QuerySpec]:
  exact = [
    QuerySpec("exact_00", "exact", "Recite the ORIGINAL_QUESTION exactly."),
    QuerySpec("exact_01", "exact", "What is the TICKET_ID? Output it exactly."),
    QuerySpec("exact_02", "exact", "Quote the exact string that starts with \"route-scale\"."),
    QuerySpec("exact_03", "exact", "What is the OWNER? Output it exactly."),
    QuerySpec("exact_04", "exact", "What is the example list for demos? Output it exactly as written."),
    QuerySpec("exact_05", "exact", "What is the launch date mentioned? Output it exactly."),
    QuerySpec("exact_06", "exact", "Return the CODE_SNIPPET function name only (exact)."),
    QuerySpec("exact_07", "exact", "In one token, answer: What algorithm is discussed?"),
    QuerySpec("exact_08", "exact", "Repeat the word after \"RISK:\" exactly."),
    QuerySpec("exact_09", "exact", "What is the mitigation described? Output the phrase exactly."),
  ]

  semantic = [
    QuerySpec("sem_00", "semantic", "Summarize the main decision in one sentence."),
    QuerySpec("sem_01", "semantic", "Why is worst-case behavior a risk?"),
    QuerySpec("sem_02", "semantic", "How does the mitigation address the risk?"),
    QuerySpec("sem_03", "semantic", "What does the code do at a high level?"),
    QuerySpec("sem_04", "semantic", "Explain the pivot choice used in the snippet."),
    QuerySpec("sem_05", "semantic", "Does the quicksort return a new list or sort in-place?"),
    QuerySpec("sem_06", "semantic", "What are the three partitions called in the snippet?"),
    QuerySpec("sem_07", "semantic", "What is the example list used for?"),
    QuerySpec("sem_08", "semantic", "What is the intent of the filler paragraphs?"),
    QuerySpec("sem_09", "semantic", "What is the role of the OWNER field?"),
    QuerySpec("sem_10", "semantic", "What should be avoided according to the decisions?"),
    QuerySpec("sem_11", "semantic", "Give a short explanation of randomized pivot here."),
    QuerySpec("sem_12", "semantic", "What is the algorithmic complexity risk described?"),
    QuerySpec("sem_13", "semantic", "What is the approximate topic of this memory log?"),
    QuerySpec("sem_14", "semantic", "Is the code meant to be optimized or educational?"),
    QuerySpec("sem_15", "semantic", "What is the relationship between left/mid/right in the code?"),
    QuerySpec("sem_16", "semantic", "What does the base-case condition do?"),
    QuerySpec("sem_17", "semantic", "What is the return statement combining?"),
    QuerySpec("sem_18", "semantic", "What does the snippet do with duplicate elements?"),
    QuerySpec("sem_19", "semantic", "What kind of input does the snippet expect?"),
  ]

  paraphrase = [
    QuerySpec("para_00", "paraphrase", "Repeat the original user question verbatim."),
    QuerySpec("para_01", "paraphrase", "Give the ticket identifier exactly."),
    QuerySpec("para_02", "paraphrase", "What exact quote contains route-scale?"),
    QuerySpec("para_03", "paraphrase", "Who owns this? Answer with the owner string only."),
    QuerySpec("para_04", "paraphrase", "State whether the sort is in-place or returns a fresh list."),
    QuerySpec("para_05", "paraphrase", "Explain in one sentence what the quicksort implementation is doing."),
    QuerySpec("para_06", "paraphrase", "How is the pivot selected?"),
    QuerySpec("para_07", "paraphrase", "What is the risk noted and the proposed mitigation?"),
    QuerySpec("para_08", "paraphrase", "What is the demo list mentioned?"),
    QuerySpec("para_09", "paraphrase", "Why do we have filler text here?"),
  ]

  distractor = [
    QuerySpec("neg_00", "distractor", "What is the CEO's name?"),
    QuerySpec("neg_01", "distractor", "What is the meeting location?"),
    QuerySpec("neg_02", "distractor", "What GPU model is being used?"),
    QuerySpec("neg_03", "distractor", "What is the password to the system?"),
    QuerySpec("neg_04", "distractor", "What is the capital of France?"),
    QuerySpec("neg_05", "distractor", "What was the weather yesterday?"),
    QuerySpec("neg_06", "distractor", "What is the SHA256 of the dataset?"),
    QuerySpec("neg_07", "distractor", "Who won the 2025 World Series?"),
    QuerySpec("neg_08", "distractor", "What is the router aux loss alpha?"),
    QuerySpec("neg_09", "distractor", "What is the company stock price?"),
  ]

  out = exact + semantic + paraphrase + distractor
  if len(out) != 50:
    raise RuntimeError(f"expected 50 queries (got {len(out)})")
  return out


def _prompt_query(*, memory: str, q: QuerySpec) -> tuple[bytes, bytes]:
  memory_text = "MEMORY:\n" + memory.strip() + "\n\n"
  query_text = (
    "INSTRUCTIONS:\n"
    "- Answer using only MEMORY.\n"
    "- If the answer is not in MEMORY, output exactly: I don't know.\n"
    "- For exact/quote requests, output exactly and nothing else.\n"
    "\n"
    f"USER:\n{q.question}\n"
    "ASSISTANT:\n"
  )
  return memory_text.encode("utf-8"), query_text.encode("utf-8")


def _producer_cfg(*, enc) -> ProducerConfig:
  return ProducerConfig(
    k=8,
    n_samples=1,
    rng_seed=0,
    temperature=1.0,
    teacher_id="checkpoint_teacher",
    teacher_dtype="bf16",
    teacher_vocab_hash="",
    masked_token_ids=masked_token_ids_from_tokenizer(enc),
  )


@torch.no_grad()
def _teacher_answer_text(
  *,
  model: Transformer,
  enc,
  memory_bytes: bytes,
  query_bytes: bytes,
  max_new_tokens: int,
  eos_token_id: int,
) -> str:
  prompt_text = (memory_bytes + query_bytes).decode("utf-8")
  prompt_ids = list(map(int, enc.encode_ordinary(prompt_text)))
  out_ids = _generate_greedy(model=model, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
  full_text = enc.decode(out_ids)
  return full_text[len(prompt_text) :]


@torch.no_grad()
def _teacher_logits(
  *,
  model: Transformer,
  input_ids: list[int],
) -> torch.Tensor:
  device = next(model.parameters()).device
  toks = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
  return model(toks)


def _train_rank(
  *,
  base_model: Transformer,
  enc,
  artifacts: list[DistillArtifact],
  student_token_ids: list[list[int]],
  rank: int,
  steps: int,
  lr: float,
) -> tuple[Transformer, float]:
  model = base_model
  model.train()
  model.requires_grad_(False)
  _set_requires_grad_for_moe_backward(model)

  apply_ldora(model, rank=int(rank), filter_fn=_ldora_filter, freeze_base=True)
  trainable = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))]
  if not trainable:
    raise RuntimeError("no trainable adapter params")

  opt = torch.optim.AdamW(trainable, lr=float(lr), betas=(0.9, 0.999), weight_decay=0.0)

  final_loss = float("nan")
  for step in range(int(steps)):
    idx = step % len(artifacts)
    toks = torch.tensor(student_token_ids[idx], dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    model.zero_grad(set_to_none=True)
    logits = model(toks)
    loss = sparse_first_token_distill_loss(student_logits=logits, artifacts=[artifacts[idx]], student_tokenizer=enc)
    loss.backward()
    opt.step()
    final_loss = float(loss.item())
  return model, final_loss


@torch.no_grad()
def _eval_rank(
  *,
  model: Transformer,
  enc,
  memory: str,
  queries: list[QuerySpec],
  teacher_answers: dict[str, str],
  max_new_tokens: int,
  eos_token_id: int,
) -> dict[str, float]:
  exact_total = 0
  exact_ok = 0
  f1_sum = 0.0
  f1_n = 0
  neg_total = 0
  neg_refusal = 0

  model.eval()
  for q in queries:
    memory_bytes, query_bytes = _prompt_query(memory=memory, q=q)
    del memory_bytes
    prompt_text = query_bytes.decode("utf-8")
    prompt_ids = list(map(int, enc.encode_ordinary(prompt_text)))
    out_ids = _generate_greedy(model=model, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
    out_text = enc.decode(out_ids)[len(prompt_text) :]

    ref = teacher_answers[q.qid]

    if q.kind == "exact":
      exact_total += 1
      exact_ok += int(out_text == ref)
    elif q.kind in ("semantic", "paraphrase"):
      f1_sum += _f1(out_text, ref)
      f1_n += 1
    else:
      neg_total += 1
      neg_refusal += int(_is_refusal(out_text))

  return {
    "exact_match_rate": (exact_ok / max(1, exact_total)),
    "semantic_f1": (f1_sum / max(1, f1_n)),
    "distractor_refusal_rate": (neg_refusal / max(1, neg_total)),
  }


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Phase 1a+1c: multi-query + rank sweep with real teacher checkpoint (pod-only).")
  p.add_argument("config", help="Path to model TOML config (moonlet/moonlight).")
  p.add_argument("--teacher_iter_dir", required=True, help="Checkpoint iteration dir containing rd.pt + dp_rank_000.pt.")
  p.add_argument("--memory_tokens", type=int, default=2048, help="Target memory length in tokens.")
  p.add_argument("--ranks", type=str, default="16,32,64,128", help="Comma-separated L/DoRA ranks.")
  p.add_argument("--steps", type=int, default=400, help="Train steps per rank (cycles through artifacts).")
  p.add_argument("--lr", type=float, default=1e-3, help="Adapter LR.")
  p.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens for teacher/student generation.")
  p.add_argument("--dtype", type=str, default="bf16", help="Model dtype for teacher+student (bf16 recommended).")
  args = p.parse_args(argv)

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required; run this on the debug pod.")

  try:
    import tiktoken  # type: ignore
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError("tiktoken is required; use the repo's container images / debug pod.") from e

  cfg = _load_cfg(args.config, dtype=args.dtype)
  enc = tiktoken.get_encoding(cfg.tokenizer)

  memory = _build_memory_text(enc=enc, target_tokens=int(args.memory_tokens))
  queries = _build_queries()

  dense_sd, expert_sd = _load_model_weights(args.teacher_iter_dir)

  # Teacher model (frozen).
  teacher = Transformer(cfg)
  teacher.load_state_dict(dense_sd, strict=False)
  teacher.load_state_dict(expert_sd, strict=False)
  teacher = teacher.cuda()
  teacher.eval()
  teacher.requires_grad_(False)

  prod_cfg = _producer_cfg(enc=enc)

  teacher_answers: dict[str, str] = {}
  artifacts: list[DistillArtifact] = []
  student_token_ids: list[list[int]] = []

  for q in queries:
    memory_bytes, query_bytes = _prompt_query(memory=memory, q=q)
    ans_text = _teacher_answer_text(
      model=teacher,
      enc=enc,
      memory_bytes=memory_bytes,
      query_bytes=query_bytes,
      max_new_tokens=int(args.max_new_tokens),
      eos_token_id=int(cfg.eos_token_id),
    )
    teacher_answers[q.qid] = ans_text

    answer_bytes = ans_text.encode("utf-8")
    full_text = (memory_bytes + query_bytes + answer_bytes).decode("utf-8")
    full_input_ids = list(map(int, enc.encode_ordinary(full_text)))

    logits = _teacher_logits(model=teacher, input_ids=full_input_ids)
    art = produce_memory_query_answer_artifact_from_logits(
      memory_bytes=memory_bytes,
      query_bytes=query_bytes,
      answer_bytes=answer_bytes,
      full_input_ids=full_input_ids,
      full_offsets=_offsets_from_tiktoken(enc, full_text, full_input_ids),
      teacher_logits=logits,
      teacher_tokenizer=enc,
      cfg=prod_cfg,
      distill_query=False,
    )
    artifacts.append(art)
    student_token_ids.append(list(map(int, enc.encode_ordinary(art.x_bytes.decode("utf-8")))))

  print(f"Built {len(artifacts)} artifacts.", flush=True)

  ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
  rows: list[tuple[int, float, float, float, float]] = []

  for r in ranks:
    # Fresh base model per rank (reusing already-loaded state dicts).
    student = Transformer(cfg)
    student.load_state_dict(dense_sd, strict=False)
    student.load_state_dict(expert_sd, strict=False)
    student = student.cuda()

    trained, final_loss = _train_rank(
      base_model=student,
      enc=enc,
      artifacts=artifacts,
      student_token_ids=student_token_ids,
      rank=int(r),
      steps=int(args.steps),
      lr=float(args.lr),
    )

    metrics = _eval_rank(
      model=trained,
      enc=enc,
      memory=memory,
      queries=queries,
      teacher_answers=teacher_answers,
      max_new_tokens=int(args.max_new_tokens),
      eos_token_id=int(cfg.eos_token_id),
    )
    rows.append((int(r), metrics["exact_match_rate"], metrics["semantic_f1"], metrics["distractor_refusal_rate"], final_loss))

    # Free CUDA memory before next rank.
    del trained
    torch.cuda.empty_cache()

  print("")
  print("Rank | Exact Match | Semantic F1 | Distractor Refusal | Loss")
  print("-----|-------------|-------------|--------------------|---------")
  for r, em, f1, neg, loss in rows:
    print(f"{r:4d} | {em:11.3f} | {f1:11.3f} | {neg:18.3f} | {loss:.6f}")

  return 0


def _offsets_from_tiktoken(enc, text: str, ids: list[int]) -> list[tuple[int | None, int | None]]:
  decoded, starts = enc.decode_with_offsets(ids)
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


if __name__ == "__main__":
  raise SystemExit(main())


"""Optimizer construction, LR scheduling, and training step for MoE.

Three optimizer types:
- NorMuon: 2D weight matrices (attention, MLP) - Polar Express orthogonalization
- AdamW: embeddings, norms, biases, router - standard AdamW
- ExpertAdamW: MoE expert MLPs (sharded via RDEP, fused with quantization)

Precision support: bf16, fp8, nvfp4 (default: nvfp4)
"""
import math
import torch
import torch.distributed as dist

from nmoe.config import Config
from nmoe import zero2
from nmoe.ks2d import KS2D

# Lazy import for normuon CUDA kernel
_muon_ext = None
def _get_muon_ext():
  global _muon_ext
  if _muon_ext is None:
    from nmoe.csrc.opt import muon as _muon_ext
  return _muon_ext


# Lazy import for SM100-only rdep extension. Dense/bf16 runs should not depend on it.
_rdep_ext = None
def _get_rdep_ext():
  global _rdep_ext
  if _rdep_ext is None:
    from nmoe.csrc import rdep as _rdep_ext
  return _rdep_ext


class NorMuon(torch.optim.Optimizer):
  """NorMuon optimizer for 2D weight matrices.

  Matches modded-nanogpt's NorMuon:
  - Momentum SGD base
  - Polar Express orthogonalization (5 iterations)
  - Low-rank variance reduction (Adafactor-style)
  - Cautious weight decay

  Only use for 2D weight tensors (attention projections, MLP weights).
  Do NOT use for embeddings, norms, biases, or 1D params.
  """

  def __init__(self, params, lr=0.023, momentum=0.95, beta2=0.95, weight_decay=1.2):
    defaults = dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay)
    super().__init__(params, defaults)
    self._plan_cache = {}  # Cache muon plans by (M, N) shape

  def _get_plan(self, M: int, N: int, Bmax: int = 32):
    """Get or create a muon plan for shape (M, N)."""
    key = (M, N)
    if key not in self._plan_cache:
      ext = _get_muon_ext()
      self._plan_cache[key] = ext.plan_create(Bmax, M, N)
    return self._plan_cache[key]

  def __del__(self):
    """Clean up muon plans."""
    ext = _get_muon_ext()
    for plan in self._plan_cache.values():
      try:
        ext.plan_destroy(plan)
      except Exception:
        pass

  @torch.no_grad()
  def step(self, closure=None):
    """Perform a single optimization step."""
    if closure is not None:
      raise RuntimeError("NorMuon does not support closure")

    ext = _get_muon_ext()

    for group in self.param_groups:
      lr = group['lr']
      momentum = group['momentum']
      beta2 = group['beta2']
      wd = group['weight_decay']

      for p in group['params']:
        if p.grad is None:
          continue

        grad = p.grad
        if grad.dim() != 2:
          raise RuntimeError(f"NorMuon only supports 2D tensors, got {grad.dim()}D")

        state = self.state[p]

        # Initialize state
        if len(state) == 0:
          state['momentum_buffer'] = torch.zeros_like(grad)
          # Variance buffer along smaller dimension (like Adafactor)
          if grad.size(0) >= grad.size(1):
            state['variance_buffer'] = torch.zeros(grad.size(0), 1, device=grad.device, dtype=torch.float32)
          else:
            state['variance_buffer'] = torch.zeros(1, grad.size(1), device=grad.device, dtype=torch.float32)

        mom_buf = state['momentum_buffer']
        var_buf = state['variance_buffer']

        # 1. Momentum update
        mom_buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(mom_buf, momentum)

        # 2. Polar Express orthogonalization
        # Reshape to (1, M, N) for batched API
        M, N = update.shape
        update_3d = update.unsqueeze(0).contiguous()

        # Need BF16 for normuon kernel
        if update_3d.dtype != torch.bfloat16:
          update_3d = update_3d.bfloat16()

        plan = self._get_plan(M, N)
        ext.plan_run(plan, update_3d.data_ptr(), 1, M, N)

        update = update_3d.squeeze(0)
        if p.dtype != update.dtype:
          update = update.to(p.dtype)

        # 3. Variance reduction (Adafactor-style), matching modded-nanogpt.
        # This stabilizes Muon scaling (e.g., batch-scaled LR) by normalizing the update.
        red_dim = -1 if grad.size(0) >= grad.size(1) else -2
        v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = update.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        v_norm = v_norm_sq.sqrt_()
        var_buf.lerp_(v_mean.to(dtype=var_buf.dtype), 1 - beta2)
        step_size = var_buf.clamp_min(1e-10).rsqrt_()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
        update.mul_(final_scale.type_as(update))

        # 4. Cautious weight decay (Muon): only decay where update and param agree in sign.
        # modded-nanogpt scales this term like wd * lr^2 (wd tensor already includes lr).
        if wd > 0:
          mask = (update * p) >= 0
          p.sub_(p * mask, alpha=float(wd) * float(lr) * float(lr))

        # 5. Apply update
        p.sub_(update, alpha=float(lr))


class ExpertMuon(torch.optim.Optimizer):
  """Muon-style optimizer for MoE expert weights ([E, M, N] tensors).

  This matches the NorMuon recipe but applies polar retraction per expert matrix
  in a single batched kernel call.
  """

  emits_weight_cache = False

  def __init__(self, params, lr=3.4e-4, momentum=0.95, beta2=0.95, weight_decay=1.2):
    defaults = dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay)
    super().__init__(params, defaults)
    self._plan_cache = {}  # Cache muon plans by (Bmax, M, N) shape

  def _get_plan(self, B: int, M: int, N: int):
    key = (B, M, N)
    if key not in self._plan_cache:
      ext = _get_muon_ext()
      self._plan_cache[key] = ext.plan_create(B, M, N)
    return self._plan_cache[key]

  def __del__(self):
    ext = _get_muon_ext()
    for plan in self._plan_cache.values():
      try:
        ext.plan_destroy(plan)
      except Exception:
        pass

  @torch.no_grad()
  def step(self, closure=None):
    if closure is not None:
      raise RuntimeError("ExpertMuon does not support closure")

    ext = _get_muon_ext()

    for group in self.param_groups:
      lr = float(group['lr'])
      momentum = float(group['momentum'])
      beta2 = float(group['beta2'])
      wd = float(group['weight_decay'])

      for p in group['params']:
        if p.grad is None:
          raise RuntimeError("ExpertMuon requires all expert grads to be present")

        grad = p.grad
        if grad.dim() != 3:
          raise RuntimeError(f"ExpertMuon only supports 3D tensors, got {grad.dim()}D")
        if not (p.is_cuda and grad.is_cuda):
          raise RuntimeError("ExpertMuon requires CUDA tensors")
        if p.dtype != torch.bfloat16 or grad.dtype != torch.bfloat16:
          raise RuntimeError("ExpertMuon requires BF16 expert weights and grads")
        if not (p.is_contiguous() and grad.is_contiguous()):
          raise RuntimeError("ExpertMuon requires contiguous expert weights and grads")

        state = self.state[p]

        E, M, N = grad.shape

        if len(state) == 0:
          state['momentum_buffer'] = torch.zeros_like(grad)
          # Variance buffer along smaller dimension (Adafactor-style), per expert.
          if M >= N:
            state['variance_buffer'] = torch.zeros(E, M, 1, device=grad.device, dtype=torch.float32)
          else:
            state['variance_buffer'] = torch.zeros(E, 1, N, device=grad.device, dtype=torch.float32)

        mom_buf = state['momentum_buffer']
        var_buf = state['variance_buffer']

        # 1) Momentum update
        mom_buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(mom_buf, momentum).contiguous()

        # 2) Polar Express orthogonalization (batched over experts)
        plan = self._get_plan(int(E), int(M), int(N))
        ext.plan_run(plan, update.data_ptr(), int(E), int(M), int(N))

        # 3) Variance reduction (Adafactor-style), matching NorMuon.
        red_dim = -1 if M >= N else -2
        v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = int(update.size(red_dim))
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        v_norm = v_norm_sq.sqrt_()
        var_buf.lerp_(v_mean.to(dtype=var_buf.dtype), 1 - beta2)
        step_size = var_buf.clamp_min(1e-10).rsqrt_()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
        update.mul_(final_scale.type_as(update))

        # 4) Cautious weight decay (Muon).
        if wd > 0:
          mask = (update * p) >= 0
          p.sub_(p * mask, alpha=float(wd) * float(lr) * float(lr))

        # 5) Apply update
        p.sub_(update, alpha=float(lr))


class ExpertAdamW(torch.optim.Optimizer):
  """Fused expert AdamW that emits blockscaled weight caches during the update.

  This removes the separate post-step refresh phase (less memory traffic, fewer kernels).
  Contract: experts are BF16 parameters; caches are FP8/NVFP4.
  """

  emits_weight_cache = True

  def __init__(self, moe_modules: list[torch.nn.Module], cfg: Config):
    params: list[torch.nn.Parameter] = []
    for moe in moe_modules:
      for name in ("W1", "W3", "W2"):
        p = getattr(moe, name, None)
        if p is None:
          raise ValueError(f"MoE module missing {name}")
        params.append(p)
    defaults = {
      "lr": cfg.lr_expert,
      "betas": (cfg.adam_beta1, cfg.adam_beta2_expert),  # Higher beta2 for FP8/NVFP4 gradient noise
      "eps": cfg.adam_eps,
      "weight_decay": cfg.weight_decay,
    }
    super().__init__(params, defaults)
    self._moes = moe_modules
    self._base_seed_u32 = int(getattr(cfg, "seed", 0)) & 0xFFFFFFFF

  def _init_state(self, p: torch.Tensor) -> dict:
    state = self.state[p]
    if len(state) == 0:
      state["step"] = torch.tensor(0.0, dtype=torch.float32)
      state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
      state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    return state

  def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
    super().load_state_dict(state_dict)
    # Step tensors may be loaded onto CUDA via map_location; keep them on CPU to
    # avoid per-step device sync when computing bias corrections.
    for st in self.state.values():
      step = st.get("step", None)
      if torch.is_tensor(step) and step.is_cuda:
        st["step"] = step.cpu()

  @torch.no_grad()
  def step(self, closure=None):  # type: ignore[override]
    if closure is not None:
      raise RuntimeError("ExpertAdamW does not support closure")
    if len(self.param_groups) != 1:
      raise RuntimeError("ExpertAdamW expects a single param group")

    group = self.param_groups[0]
    lr = float(group["lr"])
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])

    # Initialize state + bump per-param step (CPU tensors, fast).
    step_t: torch.Tensor | None = None
    for p in group["params"]:
      if p.grad is None:
        raise RuntimeError("ExpertAdamW requires all expert grads to be present")
      st = self._init_state(p)
      st["step"] += 1
      if step_t is None:
        step_t = st["step"]
    assert step_t is not None
    step = int(step_t.item())
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed_u32 = (self._base_seed_u32 ^ ((rank * 0x9e3779b9) & 0xFFFFFFFF)) & 0xFFFFFFFF
    step_u32 = step & 0xFFFFFFFF

    # Bias corrections match the standard AdamW update.
    bias_correction1 = 1.0 - (beta1**step)
    bias_correction2 = 1.0 - (beta2**step)
    step_size = lr / bias_correction1
    inv_bias_correction2_sqrt = 1.0 / math.sqrt(bias_correction2)

    for moe in self._moes:
      profile = getattr(moe, "_dtype", "nvfp4")
      if profile not in ("fp8", "nvfp4"):
        raise ValueError(f"ExpertAdamW only supports fp8/nvfp4 caches. Got {profile}")
      prof_i = 0 if profile == "fp8" else 1

      W1 = moe.W1
      W3 = moe.W3
      W2 = moe.W2
      if not (W1.is_cuda and W3.is_cuda and W2.is_cuda):
        raise RuntimeError("ExpertAdamW requires CUDA tensors")
      if not (W1.dtype == torch.bfloat16 and W3.dtype == torch.bfloat16 and W2.dtype == torch.bfloat16):
        raise RuntimeError("ExpertAdamW requires BF16 expert weights")
      if W1.grad is None or W3.grad is None or W2.grad is None:
        raise RuntimeError("Missing expert grads")
      if not (W1.grad.dtype == torch.bfloat16 and W3.grad.dtype == torch.bfloat16 and W2.grad.dtype == torch.bfloat16):
        raise RuntimeError("ExpertAdamW requires BF16 expert grads")
      if not (W1.grad.is_contiguous() and W3.grad.is_contiguous() and W2.grad.is_contiguous()):
        raise RuntimeError("Expert grads must be contiguous")
      if not (W1.is_contiguous() and W3.is_contiguous() and W2.is_contiguous()):
        raise RuntimeError("Expert weights must be contiguous")

      st1 = self._init_state(W1)
      st3 = self._init_state(W3)
      st2 = self._init_state(W2)

      # Ensure cache exists (initial use or after checkpoint restore).
      if getattr(moe, "_W_cache", None) is None:
        if not hasattr(moe, "refresh_weight_cache"):
          raise RuntimeError("MoE module missing refresh_weight_cache()")
        moe.refresh_weight_cache()
      cache = moe._W_cache
      if cache is None:
        raise RuntimeError("MoE weight cache missing after refresh")

      E, H, Dff = W1.shape
      stream = torch.cuda.current_stream(W1.device)
      _get_rdep_ext().expert_adamw_step(
        prof_i,
        W1.data_ptr(), W1.grad.data_ptr(), st1["exp_avg"].data_ptr(), st1["exp_avg_sq"].data_ptr(),
        W3.data_ptr(), W3.grad.data_ptr(), st3["exp_avg"].data_ptr(), st3["exp_avg_sq"].data_ptr(),
        W2.data_ptr(), W2.grad.data_ptr(), st2["exp_avg"].data_ptr(), st2["exp_avg_sq"].data_ptr(),
        cache.W13_q.data_ptr(), cache.W13_sf_mma.data_ptr(),
        cache.W2_q.data_ptr(), cache.W2_sf_mma.data_ptr(),
        int(E), int(H), int(Dff),
        float(lr), float(beta1), float(beta2),
        float(weight_decay), float(eps),
        float(step_size), float(inv_bias_correction2_sqrt),
        seed_u32, step_u32,
        stream,
      )

    return None


def build_optimizer(model: torch.nn.Module, cfg: Config) -> tuple[torch.optim.Optimizer | None, torch.optim.Optimizer | None, list[dict]]:
  """Build optimizers: NorMuon for 2D weights, AdamW for rest, ExpertAdamW for MoE.

  Hybrid optimizer strategy (matches modded-nanogpt):
  - NorMuon: 2D weight matrices (attention, MLP) - lr=0.023, wd=1.2
  - AdamW: embeddings, norms, biases, router - lr=0.008, wd=0.005
  - ExpertAdamW: MoE expert weights (with quantization cache)

  Returns:
    (expert_optimizer, muon_optimizer, dense_groups):
      expert_optimizer: For MoE experts (None if dense-only)
      muon_optimizer: For 2D weight matrices (None if use_muon=False)
      dense_groups: For ZeRO-2 AdamW (embeds, norms, biases, router)
  """
  use_muon = getattr(cfg, "use_muon", False)
  muon_lr = getattr(cfg, "lr_muon", 0.023)
  muon_momentum = getattr(cfg, "muon_momentum", 0.95)
  muon_wd = getattr(cfg, "muon_weight_decay", 1.2)

  # Collect parameters by type
  expert_named_params: list[tuple[str, torch.nn.Parameter]] = []
  router_params: list[torch.nn.Parameter] = []
  muon_params: list[torch.nn.Parameter] = []  # 2D weights for NorMuon
  dense_params_decay: list[torch.nn.Parameter] = []
  dense_params_no_decay: list[torch.nn.Parameter] = []

  expert_ids: set[int] = set()
  if hasattr(model, "param_sets"):
    eps, _ = model.param_sets()
    expert_ids = {id(p) for p in eps}

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue

    if id(param) in expert_ids:
      expert_named_params.append((name, param))
      continue

    # Router: separate group, no decay
    if 'router' in name:
      router_params.append(param)
      continue

    # Check if this is a 2D weight eligible for NorMuon
    is_2d_weight = param.dim() == 2 and param.numel() > 1024
    is_adam_only = (
      name.endswith('.bias') or
      'norm' in name or
      'embedding' in name or
      'lm_head' in name or
      'bungee' in name
    )

    if use_muon and is_2d_weight and not is_adam_only:
      muon_params.append(param)
    elif is_adam_only or param.dim() < 2:
      dense_params_no_decay.append(param)
    else:
      dense_params_decay.append(param)

  # Build dense param groups (ZeRO-2 AdamW)
  dense_groups = []
  if dense_params_decay:
    dense_groups.append({
      'name': 'dense_decay',
      'params': dense_params_decay,
      'lr': cfg.lr_dense,
      'weight_decay': cfg.weight_decay,
    })
  if dense_params_no_decay:
    dense_groups.append({
      'name': 'dense_no_decay',
      'params': dense_params_no_decay,
      'lr': cfg.lr_dense,
      'weight_decay': 0.0,
    })
  if router_params:
    dense_groups.append({
      'name': 'router',
      'params': router_params,
      'lr': cfg.lr_router,
      'weight_decay': 0.0,
    })

  # NorMuon optimizer for 2D weights
  muon_optimizer: torch.optim.Optimizer | None = None
  if use_muon and muon_params:
    muon_optimizer = NorMuon(
      muon_params,
      lr=muon_lr,
      momentum=muon_momentum,
      beta2=0.95,
      weight_decay=muon_wd,
    )

  # Expert optimizer (MoE only)
  expert_params = [p for _, p in expert_named_params]
  expert_optimizer: torch.optim.Optimizer | None = None
  if expert_params:
    expert_opt = getattr(cfg, "expert_opt", None)
    expert_opt = str(expert_opt or "").strip().lower()
    use_blockscaled = getattr(cfg, "dtype", "nvfp4") in ("fp8", "nvfp4")
    if expert_opt in ("", "auto"):
      if use_blockscaled:
        moes: list[torch.nn.Module] = []
        for blk in getattr(model, "blocks", []):
          ffn = getattr(blk, "ffn", None)
          if ffn is not None and hasattr(ffn, "W1") and hasattr(ffn, "W3") and hasattr(ffn, "W2"):
            moes.append(ffn)
        expert_optimizer = ExpertAdamW(moes, cfg)
      else:
        expert_optimizer = torch.optim.AdamW(
          expert_params,
          lr=cfg.lr_expert,
          betas=(cfg.adam_beta1, cfg.adam_beta2_expert),
          eps=cfg.adam_eps,
          weight_decay=cfg.weight_decay,
        )
    elif expert_opt == "adamw":
      expert_optimizer = torch.optim.AdamW(
        expert_params,
        lr=cfg.lr_expert,
        betas=(cfg.adam_beta1, cfg.adam_beta2_expert),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
      )
    elif expert_opt == "muon":
      expert_optimizer = ExpertMuon(
        expert_params,
        lr=float(cfg.lr_expert),
        momentum=float(getattr(cfg, "muon_momentum", 0.95)),
        beta2=float(getattr(cfg, "adam_beta2_expert", 0.95)),
        weight_decay=float(getattr(cfg, "muon_weight_decay", 1.2)),
      )
    elif expert_opt == "ks2d":
      warmup = getattr(cfg, "ks2d_warmup_steps", None)
      warmup_steps = int(cfg.warmup_steps) if warmup is None else int(warmup)
      rank = int(getattr(cfg, "ks2d_rank", 8))
      cb_freq = int(getattr(cfg, "ks2d_codebook_update_freq", 100))
      max_rms = float(getattr(cfg, "ks2d_max_update_rms", 1.0))

      groups: dict[str, list[torch.nn.Parameter]] = {}
      for name, p in expert_named_params:
        leaf = name.rsplit(".", 1)[-1]
        role = f"expert_{leaf.lower()}" if leaf in ("W1", "W2", "W3") else "expert"
        groups.setdefault(role, []).append(p)
      param_groups = [{"ks2d_role": r, "params": ps} for r, ps in sorted(groups.items())]
      expert_optimizer = KS2D(
        param_groups,
        lr=float(cfg.lr_expert),
        momentum=float(cfg.adam_beta1),
        beta2=float(cfg.adam_beta2_expert),
        weight_decay=float(cfg.weight_decay),
        rank=rank,
        codebook_update_freq=cb_freq,
        warmup_steps=warmup_steps,
        max_update_rms=max_rms,
      )
    else:
      raise ValueError(f"Unknown expert_opt={expert_opt!r} (expected: auto|adamw|muon|ks2d)")

  return expert_optimizer, muon_optimizer, dense_groups


def update_lr(optimizer: torch.optim.Optimizer | None, muon_optimizer: torch.optim.Optimizer | None, dense_groups: list[dict], step: int, tokens_seen: int, cfg: Config) -> float:
  """Update learning rate using a DeepSeek-style WSD schedule.

  WSD (Warmup-Sustain-Decay):
  1) Warmup: linear 0 → peak over warmup_steps (by step)
  2) Sustain: hold at peak until hold_tokens (by tokens)
  3) Cosine decay: decay over decay_tokens down to floor (absolute LR)
  4) Floor: hold at floor indefinitely

  Contract:
  - cfg.decay_floor is an *absolute* LR (not a ratio).
  - The same schedule shape is applied to expert, dense, and router peaks,
    with a shared absolute floor.
  """
  peak_expert = float(cfg.lr_expert)
  peak_dense = float(cfg.lr_dense)
  peak_router = float(cfg.lr_router)
  floor = float(cfg.decay_floor)

  if floor <= 0.0:
    raise RuntimeError(f"decay_floor must be > 0 (got {floor})")
  if floor > peak_expert or floor > peak_dense:
    raise RuntimeError(
      f"decay_floor ({floor}) must be <= lr_expert ({peak_expert}) and lr_dense ({peak_dense})"
    )

  # Compute schedule scale factor (0→1 during warmup, 1 during hold, 1→floor_ratio during decay)
  if step < cfg.warmup_steps:
    lr_scale = (step + 1) / max(1, int(cfg.warmup_steps))
  elif tokens_seen < cfg.hold_tokens:
    lr_scale = 1.0
  else:
    t = int(tokens_seen - cfg.hold_tokens)
    if t < cfg.decay_tokens:
      denom = float(max(1, int(cfg.decay_tokens)))
      lr_scale = 0.5 * (1.0 + math.cos(math.pi * float(t) / denom))
    else:
      lr_scale = 0.0

  # Apply schedule to each group's peak LR (floor applied at scale=0)
  floor_expert = floor
  floor_dense = floor
  floor_router = min(floor, peak_router)  # Router floor capped at its peak
  lr_expert = floor_expert + (peak_expert - floor_expert) * lr_scale
  lr_dense = floor_dense + (peak_dense - floor_dense) * lr_scale
  lr_router = floor_router + (peak_router - floor_router) * lr_scale

  if optimizer is not None:
    for g in optimizer.param_groups:
      g['lr'] = lr_expert

  # NorMuon uses its own LR schedule (same shape, different peak)
  if muon_optimizer is not None:
    peak_muon = float(getattr(cfg, "lr_muon", 0.023))
    floor_muon = min(floor, peak_muon)
    lr_muon = floor_muon + (peak_muon - floor_muon) * lr_scale
    for g in muon_optimizer.param_groups:
      g['lr'] = lr_muon

  for g in dense_groups:
    # Router group uses lr_router, others use lr_dense
    if g.get('name') == 'router':
      g['lr'] = lr_router
    else:
      g['lr'] = lr_dense

  # Return the dense LR for logging (backward compatible with older single-LR configs).
  return float(lr_dense)


def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, muon_optimizer: torch.optim.Optimizer | None, dense_groups: list[dict], zero2_state: dict, cfg: Config, world: int) -> None:
  """Optimizer step with ZeRO-2, NorMuon, and post-step hooks.

  Handles:
  - ZeRO-2 AdamW stepping for dense params (if world > 1)
  - NorMuon stepping for 2D weights (attention, MLP)
  - ExpertAdamW stepping for MoE expert params
  - Post-step model updates (quantization rebuild, router bias)
  """
  # ZeRO-2 AdamW step for dense params (embeds, norms, biases, router)
  if world > 1:
    zero2.step_dense_adamw(
      dense_groups,
      state=zero2_state,
      pg=dist.group.WORLD,
      betas=(cfg.adam_beta1, cfg.adam_beta2),
      eps=cfg.adam_eps,
    )
  else:
    zero2.step_dense_adamw(dense_groups, state=zero2_state)

  # NorMuon step for 2D weight matrices
  if muon_optimizer is not None:
    muon_optimizer.step()

  # Expert optimizer step (no ZeRO-2, params already sharded via RDEP)
  if optimizer is not None:
    optimizer.step()

  # Bias updates should decay with router LR so they don't dominate during warmdown.
  router_bias_gamma = float(getattr(cfg, "router_bias_update_rate", 0.0))
  if router_bias_gamma != 0.0:
    peak_router = float(getattr(cfg, "lr_router", 0.0))
    floor_router = min(float(getattr(cfg, "decay_floor", 0.0)), peak_router)
    lr_router = peak_router
    for g in dense_groups:
      if g.get("name") == "router":
        lr_router = float(g.get("lr", peak_router))
        break
    denom = peak_router - floor_router
    if denom > 0.0:
      lr_scale = (lr_router - floor_router) / denom
      if lr_scale < 0.0:
        lr_scale = 0.0
      elif lr_scale > 1.0:
        lr_scale = 1.0
      router_bias_gamma *= lr_scale

  # Post-step hooks (quantization rebuild, router bias update)
  with torch.no_grad():
    for i, blk in enumerate(model.blocks):
      ffn = getattr(blk, 'ffn', None)
      if ffn is None:
        continue

      # Refresh quantized weight cache (FP8/NVFP4)
      if hasattr(ffn, 'refresh_weight_cache'):
        emits_cache = bool(getattr(optimizer, "emits_weight_cache", False)) if optimizer is not None else False
        if not emits_cache:
          try:
            ffn.refresh_weight_cache()
          except Exception as e:
            raise RuntimeError(f"post-step weight cache refresh failed (block={i}, ffn={type(ffn).__name__})") from e

      # Update router bias for aux-free load balancing
      if router_bias_gamma != 0.0 and hasattr(ffn, 'router') and getattr(ffn, 'last_loads', None) is not None:
        try:
          ffn.router.update_bias(ffn.last_loads, gamma=router_bias_gamma)
        except Exception as e:
          raise RuntimeError(f"post-step router bias update failed (block={i}, ffn={type(ffn).__name__})") from e

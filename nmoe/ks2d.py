from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class KS2DKey:
  role: str
  out_dim: int
  in_dim: int


def _key_to_str(key: KS2DKey) -> str:
  return f"{key.role}|{key.out_dim}x{key.in_dim}"


def _key_from_str(s: str) -> KS2DKey:
  role, shape = s.split("|", 1)
  out_s, in_s = shape.split("x", 1)
  return KS2DKey(role=role, out_dim=int(out_s), in_dim=int(in_s))


@dataclass
class _SideCodebook:
  Q: torch.Tensor  # [dim, k], dtype matches params (bf16)
  lam: torch.Tensor  # [k], float32
  tau: torch.Tensor  # [], float32 (trace)
  initialized: bool


@dataclass
class _Codebook:
  k: int
  left: _SideCodebook
  right: _SideCodebook


_muon_ext = None


def _get_muon_ext():
  global _muon_ext
  if _muon_ext is None:
    from nmoe.csrc.opt import muon as _muon_ext
  return _muon_ext


def _orthogonal_init(dim: int, k: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  x = torch.randn((dim, k), device=device, dtype=torch.float32)
  q, _ = torch.linalg.qr(x, mode="reduced")
  return q.to(dtype=dtype)


def _as_bmn(x: torch.Tensor) -> torch.Tensor:
  if x.ndim == 2:
    return x.unsqueeze(0)
  if x.ndim == 3:
    return x
  raise ValueError(f"Expected a 2D/3D tensor, got {x.ndim}D")


def _mean_cov_action_left(mats: Iterable[torch.Tensor], Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
  """Compute mean covariance action for the left side, using implicit matvecs.

  For matrices {M_i} (each [B_i, M, N] or [M, N]) and basis Q ([M, k]), this computes:
    Y = sum_i sum_b M_{i,b} (M_{i,b}^T Q)        [M, k]
    G = sum_i sum_b (M_{i,b}^T Q)^T (M_{i,b}^T Q) [k, k]
    tau = sum_i sum_b ||M_{i,b}||_F^2            []
    n = total number of matrices (sum B_i)

  No materialization of M M^T.
  """
  dev = Q.device
  k = int(Q.size(1))
  Y = torch.zeros((int(Q.size(0)), k), device=dev, dtype=torch.float32)
  G = torch.zeros((k, k), device=dev, dtype=torch.float32)
  tau = torch.zeros((), device=dev, dtype=torch.float32)
  n = 0

  for m in mats:
    bmn = _as_bmn(m)
    if bmn.numel() == 0:
      continue
    B = int(bmn.size(0))
    n += B

    mq = torch.matmul(bmn.transpose(1, 2), Q)  # [B, N, k]
    y_i = torch.matmul(bmn, mq)  # [B, M, k]
    Y.add_(y_i.float().sum(dim=0))

    mq_f = mq.float().reshape(-1, k)
    G.add_(mq_f.transpose(0, 1) @ mq_f)
    tau.add_(bmn.float().square().sum())

  return Y, G, tau, n


def _mean_cov_action_right(mats: Iterable[torch.Tensor], Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
  """Compute mean covariance action for the right side, using implicit matvecs.

  For matrices {M_i} (each [B_i, M, N] or [M, N]) and basis Q ([N, k]), this computes:
    Y = sum_i sum_b M_{i,b}^T (M_{i,b} Q)         [N, k]
    G = sum_i sum_b (M_{i,b} Q)^T (M_{i,b} Q)     [k, k]
    tau = sum_i sum_b ||M_{i,b}||_F^2             []
    n = total number of matrices (sum B_i)
  """
  dev = Q.device
  k = int(Q.size(1))
  Y = torch.zeros((int(Q.size(0)), k), device=dev, dtype=torch.float32)
  G = torch.zeros((k, k), device=dev, dtype=torch.float32)
  tau = torch.zeros((), device=dev, dtype=torch.float32)
  n = 0

  for m in mats:
    bmn = _as_bmn(m)
    if bmn.numel() == 0:
      continue
    B = int(bmn.size(0))
    n += B

    mq = torch.matmul(bmn, Q)  # [B, M, k]
    y_i = torch.matmul(bmn.transpose(1, 2), mq)  # [B, N, k]
    Y.add_(y_i.float().sum(dim=0))

    mq_f = mq.float().reshape(-1, k)
    G.add_(mq_f.transpose(0, 1) @ mq_f)
    tau.add_(bmn.float().square().sum())

  return Y, G, tau, n


def _all_reduce_mean_(x: torch.Tensor, n: int) -> int:
  """All-reduce x and n across WORLD; return global n.

  Contract: x is float32; n is the number of matrices contributing to x.
  """
  if not (dist.is_available() and dist.is_initialized()):
    return n
  n_t = x.new_tensor([float(n)], dtype=torch.float32)
  dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
  dist.all_reduce(x, op=dist.ReduceOp.SUM)
  n_g = int(n_t.item())
  if n_g > 0:
    x.div_(float(n_g))
  return n_g


def _invroot_apply_left(m: torch.Tensor, Q: torch.Tensor, *, a: torch.Tensor, b: float) -> torch.Tensor:
  """Apply InvRoot = b I + Q diag(a - b) Q^T on the left of m.

  Supports m as [M, N] or [B, M, N].
  """
  bmn = _as_bmn(m)
  b_t = bmn.new_tensor(float(b))
  if Q.numel() == 0:
    out = bmn.mul(b_t)
    return out.squeeze(0) if m.ndim == 2 else out

  delta = (a - float(b)).to(dtype=bmn.dtype)
  Qt_m = torch.matmul(Q.transpose(0, 1), bmn)  # [B, k, N]
  corr = delta.view(1, -1, 1) * Qt_m  # [B, k, N]
  out = bmn.mul(b_t)
  out.add_(torch.matmul(Q, corr))
  return out.squeeze(0) if m.ndim == 2 else out


def _invroot_apply_right(m: torch.Tensor, Q: torch.Tensor, *, a: torch.Tensor, b: float) -> torch.Tensor:
  """Apply InvRoot = b I + Q diag(a - b) Q^T on the right of m.

  Supports m as [M, N] or [B, M, N].
  """
  bmn = _as_bmn(m)
  b_t = bmn.new_tensor(float(b))
  if Q.numel() == 0:
    out = bmn.mul(b_t)
    return out.squeeze(0) if m.ndim == 2 else out

  delta = (a - float(b)).to(dtype=bmn.dtype)
  mQ = torch.matmul(bmn, Q)  # [B, M, k]
  corr = mQ * delta.view(1, 1, -1)  # [B, M, k]
  out = bmn.mul(b_t)
  out.add_(torch.matmul(corr, Q.transpose(0, 1)))
  return out.squeeze(0) if m.ndim == 2 else out


def _rms_clip_(x: torch.Tensor, max_rms: float, eps: float) -> None:
  if max_rms <= 0.0:
    raise ValueError(f"max_rms must be > 0 (got {max_rms})")
  bmn = _as_bmn(x)
  rms = bmn.float().square().mean(dim=(-2, -1), keepdim=True).sqrt_()
  scale = (float(max_rms) / rms.clamp_min_(eps)).clamp(max=1.0)
  bmn.mul_(scale.to(dtype=bmn.dtype))


class KS2D(torch.optim.Optimizer):
  """KitchenSink 2D optimizer (KS2D): compressed PSGD preconditioning + polar retraction + post-adaptivity.

  Contract:
  - Supports 2D weights ([M, N]) and batched 2D weights ([B, M, N]) (e.g., experts).
  - Param groups must set `ks2d_role` (string).
  - Persistent per-param state is exactly two tensors: m and v (AdamW-class).
  - Codebooks are shared per (role, out_dim, in_dim) and updated amortized using S=m.
  """

  emits_weight_cache = False

  def __init__(
    self,
    params: Any,
    *,
    lr: float,
    momentum: float,
    beta2: float,
    weight_decay: float,
    rank: int = 8,
    codebook_update_freq: int = 100,
    warmup_steps: int = 500,
    max_update_rms: float = 1.0,
    alpha: float = 0.25,
    eps: float = 1e-8,
    lambda_floor: float = 1e-12,
    codebook_beta: float | None = None,
    muon_steps: int = 5,
    muon_coeff_mode: int = 1,
  ):
    if lr <= 0.0:
      raise ValueError(f"lr must be > 0 (got {lr})")
    if not 0.0 <= momentum < 1.0:
      raise ValueError(f"momentum must be in [0,1) (got {momentum})")
    if not 0.0 <= beta2 < 1.0:
      raise ValueError(f"beta2 must be in [0,1) (got {beta2})")
    if weight_decay < 0.0:
      raise ValueError(f"weight_decay must be >= 0 (got {weight_decay})")
    if rank <= 0:
      raise ValueError(f"rank must be > 0 (got {rank})")
    if codebook_update_freq <= 0:
      raise ValueError(f"codebook_update_freq must be > 0 (got {codebook_update_freq})")
    if warmup_steps < 0:
      raise ValueError(f"warmup_steps must be >= 0 (got {warmup_steps})")
    if max_update_rms <= 0.0:
      raise ValueError(f"max_update_rms must be > 0 (got {max_update_rms})")
    if not (0.0 < alpha <= 0.5):
      raise ValueError(f"alpha must be in (0, 0.5] (got {alpha})")
    if eps <= 0.0:
      raise ValueError(f"eps must be > 0 (got {eps})")
    if lambda_floor < 0.0:
      raise ValueError(f"lambda_floor must be >= 0 (got {lambda_floor})")
    if muon_steps <= 0:
      raise ValueError(f"muon_steps must be > 0 (got {muon_steps})")

    defaults = dict(
      lr=float(lr),
      momentum=float(momentum),
      beta2=float(beta2),
      weight_decay=float(weight_decay),
    )
    super().__init__(params, defaults)

    self._rank = int(rank)
    self._codebook_update_freq = int(codebook_update_freq)
    self._warmup_steps = int(warmup_steps)
    self._max_update_rms = float(max_update_rms)
    self._alpha = float(alpha)
    self._eps = float(eps)
    self._lambda_floor = float(lambda_floor)
    self._codebook_beta = float(beta2 if codebook_beta is None else codebook_beta)
    self._muon_steps = int(muon_steps)
    self._muon_coeff_mode = int(muon_coeff_mode)

    self._step = 0
    self._codebooks: dict[KS2DKey, _Codebook] = {}
    self._plan_cache: dict[tuple[int, int, int], int] = {}  # (Bmax, M, N) -> plan

    for group in self.param_groups:
      role = group.get("ks2d_role", None)
      if not isinstance(role, str) or not role:
        raise ValueError("KS2D requires param groups to set a non-empty `ks2d_role` string")
      for p in group.get("params", []):
        if not isinstance(p, torch.Tensor):
          continue
        if p.ndim not in (2, 3):
          raise ValueError(f"KS2D only supports 2D/3D params; got {tuple(p.shape)} for role={role}")

  def _get_plan(self, B: int, M: int, N: int) -> int:
    if B <= 0:
      raise ValueError(f"Invalid batch size B={B}")
    Bmax = int(B)
    key = (Bmax, M, N)
    if key not in self._plan_cache:
      ext = _get_muon_ext()
      self._plan_cache[key] = int(ext.plan_create(Bmax, M, N))
    return self._plan_cache[key]

  def __del__(self):
    try:
      ext = _get_muon_ext()
    except Exception:
      return
    for plan in self._plan_cache.values():
      try:
        ext.plan_destroy(int(plan))
      except Exception:
        pass

  def state_dict(self) -> dict:  # type: ignore[override]
    sd = super().state_dict()
    sd["ks2d"] = {
      "step": int(self._step),
      "rank": int(self._rank),
      "codebook_update_freq": int(self._codebook_update_freq),
      "warmup_steps": int(self._warmup_steps),
      "max_update_rms": float(self._max_update_rms),
      "alpha": float(self._alpha),
      "eps": float(self._eps),
      "lambda_floor": float(self._lambda_floor),
      "codebook_beta": float(self._codebook_beta),
      "muon_steps": int(self._muon_steps),
      "muon_coeff_mode": int(self._muon_coeff_mode),
      "codebooks": {
        _key_to_str(k): {
          "k": int(cb.k),
          "left": {
            "Q": cb.left.Q,
            "lam": cb.left.lam,
            "tau": cb.left.tau,
            "initialized": bool(cb.left.initialized),
          },
          "right": {
            "Q": cb.right.Q,
            "lam": cb.right.lam,
            "tau": cb.right.tau,
            "initialized": bool(cb.right.initialized),
          },
        }
        for k, cb in self._codebooks.items()
      },
    }
    return sd

  def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
    ks2d = state_dict.get("ks2d", None)
    super().load_state_dict({k: v for k, v in state_dict.items() if k != "ks2d"})
    if ks2d is None:
      return
    self._step = int(ks2d.get("step", 0))
    self._rank = int(ks2d.get("rank", self._rank))
    self._codebook_update_freq = int(ks2d.get("codebook_update_freq", self._codebook_update_freq))
    self._warmup_steps = int(ks2d.get("warmup_steps", self._warmup_steps))
    self._max_update_rms = float(ks2d.get("max_update_rms", self._max_update_rms))
    self._alpha = float(ks2d.get("alpha", self._alpha))
    self._eps = float(ks2d.get("eps", self._eps))
    self._lambda_floor = float(ks2d.get("lambda_floor", self._lambda_floor))
    self._codebook_beta = float(ks2d.get("codebook_beta", self._codebook_beta))
    self._muon_steps = int(ks2d.get("muon_steps", self._muon_steps))
    self._muon_coeff_mode = int(ks2d.get("muon_coeff_mode", self._muon_coeff_mode))

    cbs = ks2d.get("codebooks", {})
    codebooks: dict[KS2DKey, _Codebook] = {}
    for k_str, v in cbs.items():
      k = _key_from_str(k_str)
      left = v["left"]
      right = v["right"]
      codebooks[k] = _Codebook(
        k=int(v["k"]),
        left=_SideCodebook(
          Q=left["Q"],
          lam=left["lam"],
          tau=left["tau"],
          initialized=bool(left.get("initialized", True)),
        ),
        right=_SideCodebook(
          Q=right["Q"],
          lam=right["lam"],
          tau=right["tau"],
          initialized=bool(right.get("initialized", True)),
        ),
      )
    self._codebooks = codebooks

  def _get_or_create_codebook(self, key: KS2DKey, *, device: torch.device, dtype: torch.dtype) -> _Codebook:
    cb = self._codebooks.get(key, None)
    if cb is not None:
      return cb
    k = min(self._rank, key.out_dim, key.in_dim)
    QL = _orthogonal_init(key.out_dim, k, device=device, dtype=dtype)
    QR = _orthogonal_init(key.in_dim, k, device=device, dtype=dtype)
    lam0 = torch.full((k,), 1.0, device=device, dtype=torch.float32)
    tau0 = torch.full((), float(k), device=device, dtype=torch.float32)
    cb = _Codebook(
      k=k,
      left=_SideCodebook(Q=QL, lam=lam0.clone(), tau=tau0.clone(), initialized=False),
      right=_SideCodebook(Q=QR, lam=lam0.clone(), tau=tau0.clone(), initialized=False),
    )
    self._codebooks[key] = cb
    return cb

  def _update_codebook_for_group(self, key: KS2DKey, cb: _Codebook, mats: list[torch.Tensor]) -> None:
    beta_cb = float(self._codebook_beta)
    YL, GL, tauL, nL = _mean_cov_action_left(mats, cb.left.Q)
    YR, GR, tauR, nR = _mean_cov_action_right(mats, cb.right.Q)
    if nL != nR:
      raise RuntimeError(f"KS2D internal error: left/right mat counts differ ({nL} vs {nR}) for {key}")

    n_g = _all_reduce_mean_(YL, nL)
    _all_reduce_mean_(GL, nL)
    _all_reduce_mean_(tauL, nL)
    _all_reduce_mean_(YR, nR)
    _all_reduce_mean_(GR, nR)
    _all_reduce_mean_(tauR, nR)
    if n_g <= 0:
      return

    # Subspace update (block power + QR) and EMA of scalar stats.
    if float(YL.norm().item()) > 0.0:
      qL, _ = torch.linalg.qr(YL, mode="reduced")
      cb.left.Q.copy_(qL.to(dtype=cb.left.Q.dtype))
      cb.left.initialized = True
    if float(YR.norm().item()) > 0.0:
      qR, _ = torch.linalg.qr(YR, mode="reduced")
      cb.right.Q.copy_(qR.to(dtype=cb.right.Q.dtype))
      cb.right.initialized = True

    lamL_new = torch.diag(GL).clamp_min_(0.0)
    lamR_new = torch.diag(GR).clamp_min_(0.0)
    cb.left.lam.mul_(beta_cb).add_(lamL_new, alpha=(1.0 - beta_cb))
    cb.right.lam.mul_(beta_cb).add_(lamR_new, alpha=(1.0 - beta_cb))
    cb.left.tau.mul_(beta_cb).add_(tauL, alpha=(1.0 - beta_cb))
    cb.right.tau.mul_(beta_cb).add_(tauR, alpha=(1.0 - beta_cb))

  def _maybe_update_codebooks(self) -> None:
    if (self._step % self._codebook_update_freq) != 0:
      return
    for group in self.param_groups:
      role = str(group["ks2d_role"])
      params = [p for p in group.get("params", []) if isinstance(p, torch.Tensor) and p.grad is not None]
      if not params:
        continue
      p0 = params[0]
      out_dim, in_dim = int(p0.size(-2)), int(p0.size(-1))
      key = KS2DKey(role=role, out_dim=out_dim, in_dim=in_dim)
      cb = self._get_or_create_codebook(key, device=p0.device, dtype=p0.dtype)
      mats = [self.state[p]["m"] for p in params]
      self._update_codebook_for_group(key, cb, mats)

  @torch.no_grad()
  def step(self, closure=None):  # type: ignore[override]
    if closure is not None:
      raise RuntimeError("KS2D does not support closure")

    ext = _get_muon_ext()
    self._step += 1
    step = int(self._step)

    # Pass 1: init state and momentum update (S=m) for all params.
    for group in self.param_groups:
      beta1 = float(group["momentum"])
      for p in group.get("params", []):
        if not isinstance(p, torch.Tensor):
          continue
        if p.grad is None:
          raise RuntimeError("KS2D requires all grads to be present for its params")
        st = self.state[p]
        if len(st) == 0:
          st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        st["m"].lerp_(p.grad, 1.0 - beta1)

    # Pass 2: amortized codebook update (uses updated momentums).
    self._maybe_update_codebooks()

    # Pass 3: apply updates.
    for group in self.param_groups:
      lr = float(group["lr"])
      beta2 = float(group["beta2"])
      wd = float(group["weight_decay"])
      role = str(group["ks2d_role"])

      params = [p for p in group.get("params", []) if isinstance(p, torch.Tensor)]
      if not params:
        continue

      # Warmup: plain AdamW (m / sqrt(v)).
      if step <= self._warmup_steps:
        for p in params:
          if p.grad is None:
            raise RuntimeError("KS2D requires all grads to be present for its params")
          st = self.state[p]
          m = st["m"]
          v = st["v"]
          v.lerp_(p.grad.square(), 1.0 - beta2)
          denom = v.sqrt().add_(self._eps)
          update = m / denom
          _rms_clip_(update, self._max_update_rms, self._eps)
          if wd != 0.0:
            p.mul_(1.0 - lr * wd)
          p.sub_(update, alpha=lr)
        continue

      p0 = params[0]
      out_dim, in_dim = int(p0.size(-2)), int(p0.size(-1))
      key = KS2DKey(role=role, out_dim=out_dim, in_dim=in_dim)
      cb = self._get_or_create_codebook(key, device=p0.device, dtype=p0.dtype)
      if not (cb.left.initialized and cb.right.initialized):
        mats = [self.state[p]["m"] for p in params]
        self._update_codebook_for_group(key, cb, mats)

      k = cb.k
      eps = float(self._eps)
      alpha = float(self._alpha)
      lf = float(self._lambda_floor)

      lamL = cb.left.lam[:k]
      tauL = cb.left.tau
      denomL = float(max(1, out_dim - k))
      lam_bar_L = ((tauL - lamL.sum()) / denomL).clamp_min(lf)
      aL = (lamL + eps).pow(-alpha)
      bL = float((lam_bar_L + eps).pow(-alpha).item())

      lamR = cb.right.lam[:k]
      tauR = cb.right.tau
      denomR = float(max(1, in_dim - k))
      lam_bar_R = ((tauR - lamR.sum()) / denomR).clamp_min(lf)
      aR = (lamR + eps).pow(-alpha)
      bR = float((lam_bar_R + eps).pow(-alpha).item())

      for p in params:
        st = self.state[p]
        m = st["m"]
        v = st["v"]

        d0 = _invroot_apply_left(m, cb.left.Q, a=aL, b=bL)
        d0 = _invroot_apply_right(d0, cb.right.Q, a=aR, b=bR)
        if not d0.is_cuda:
          raise RuntimeError("KS2D requires CUDA tensors (no CPU path)")
        if d0.dtype != torch.bfloat16:
          raise RuntimeError(f"KS2D requires BF16 tensors (got {d0.dtype})")

        # Retract (polar factor) and Frobenius-norm match (per matrix).
        d0_bmn = _as_bmn(d0).contiguous()
        B = int(d0_bmn.size(0))
        d0_norm = d0_bmn.float().norm(dim=(-2, -1), keepdim=True).clamp_min_(self._eps)

        plan = self._get_plan(B, out_dim, in_dim)
        ext.plan_run(
          int(plan),
          int(d0_bmn.data_ptr()),
          B,
          out_dim,
          in_dim,
          self._muon_steps,
          self._muon_coeff_mode,
        )
        u_norm = d0_bmn.float().norm(dim=(-2, -1), keepdim=True).clamp_min_(self._eps)
        d0_bmn.mul_((d0_norm / u_norm).to(dtype=d0_bmn.dtype))
        d1 = d0_bmn.squeeze(0) if d0.ndim == 2 else d0_bmn

        # Post-transform adaptivity.
        v.lerp_(d1.square(), 1.0 - beta2)
        denom = v.sqrt().add_(self._eps)
        update = d1 / denom
        _rms_clip_(update, self._max_update_rms, self._eps)

        if wd != 0.0:
          p.mul_(1.0 - lr * wd)
        p.sub_(update, alpha=lr)

    return None


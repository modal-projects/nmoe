"""PERL (Parameter‑Efficient RL) utilities.

Phase 1 scope:
- L/DoRA Mode A (frozen g, B0=0, rank-independent A0 init)
- IRC computation helpers (parameter-space diagnostics)
- Generic patching utilities for nn.Linear modules
"""

from nmoe.perl.ldora import LDoRALinear
from nmoe.perl.apply import apply_ldora
from nmoe.perl.irc import (
  IrcThresholds,
  IrcSummary,
  compute_irc_summary,
)
from nmoe.perl.policy import validate_optimizer_contract

__all__ = [
  "LDoRALinear",
  "apply_ldora",
  "IrcThresholds",
  "IrcSummary",
  "compute_irc_summary",
  "validate_optimizer_contract",
]

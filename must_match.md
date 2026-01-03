• Must‑Match Semantics Checklist (Inference)
  Scope: enough to reproduce teacher logits for offline distillation. Everything below is “shape + ordering + placement” sensitive.

  I’ll use:

  - B batch, S seq, D hidden_size
  - H num_attention_heads, KV num_key_value_heads, Dh head_dim
  - G = H / KV (an integer)
  - E num_local_experts, K num_experts_per_tok (top‑k)

  ———

  ## 0) Shared transformer block contract (GLM‑4.7 and MiniMax‑M2)

  Both are pre‑norm blocks:

  1. resid = x  shape [B,S,D]
  2. x = RMSNorm(x)  [B,S,D]
  3. x = SelfAttn(x, position_embeddings, attention_mask, cache?)  [B,S,D]
  4. x = resid + x  [B,S,D]
  5. resid = x  [B,S,D]
  6. x = RMSNorm(x)  [B,S,D]
  7. x = MLP_or_MoE(x)  [B,S,D]
  8. x = resid + x  [B,S,D]

  nmoe already matches this structure; deltas are inside attention + MoE.

  ———

  ## 1) GLM‑4.7: Attention (GQA + per‑head QK norm + partial RoPE)

  Config highlights: D=5120, H=96, KV=8, Dh=128, attention_bias=True, use_qk_norm=True, partial_rotary_factor=0.5.

  ### 1.1 Projections (+ bias semantics)

  Input: x [B,S,D]

  - q = q_proj(x) with bias → [B,S,H*Dh]
  - k = k_proj(x) with bias → [B,S,KV*Dh]
  - v = v_proj(x) with bias → [B,S,KV*Dh]

  Then reshape before QK norm:

  - q = q.view(B,S,H,Dh)  [B,S,H,Dh]
  - k = k.view(B,S,KV,Dh)  [B,S,KV,Dh]
  - v = v.view(B,S,KV,Dh)  [B,S,KV,Dh]

  ### 1.2 QK norm placement (GLM‑specific)

  Per-head last-dim RMSNorm(head_dim) applied on the shaped tensors:

  - q = RMSNorm(Dh)(q)  [B,S,H,Dh]
  - k = RMSNorm(Dh)(k)  [B,S,KV,Dh]
  - (v is not normalized)

  This is not the MiniMax variant and not MLA’s LoRA bottleneck norms.

  ### 1.3 Transpose + RoPE application

  Transpose to attention layout:

  - q = q.transpose(1,2) → [B,H,S,Dh]
  - k = k.transpose(1,2) → [B,KV,S,Dh]
  - v = v.transpose(1,2) → [B,KV,S,Dh]

  RoPE:

  - GLM computes rotary dim: Dr = int(Dh * partial_rotary_factor) = 64.
  - It applies apply_rotary_pos_emb by splitting last dim:
      - rotate [..., :Dr], pass-through [..., Dr:].

  Position embeddings contract

  - position_embeddings = (cos, sin) with cos/sin shaped [B,S,Dr].
  - In apply, they do cos = cos.unsqueeze(1) to broadcast to [B,1,S,Dr].
    So your producer must generate cos/sin with that broadcast behavior.

  ### 1.4 GQA expansion and attention compute

  - Expand KV across groups: k_rep = repeat_kv(k, n_rep=G) → [B,H,S,Dh] and same for v.
  - Attention is causal softmax with scaling Dh**-0.5 and dropout=0 in eval:
      - output o [B,H,S,Dh].

  Finally:

  - o = o.transpose(1,2).reshape(B,S,H*Dh) → [B,S,H*Dh]
  - y = o_proj(o) (no bias) → [B,S,D]

  ———

  ## 2) GLM‑4.7: MoE (sigmoid routing + group mask + norm + post-scale + shared experts)

  Config highlights: n_routed_experts=160, num_experts_per_tok=8, n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5, n_shared_experts=1,
  moe_intermediate_size=1536, first_k_dense_replace=3.

  ### 2.1 Dense vs MoE switch

  - If layer_idx < first_k_dense_replace: use dense MLP (SwiGLU: act(gate_proj)*up_proj -> down_proj).
  - Else: use MoE.

  ### 2.2 Router logits (matmul weight; no bias term)

  Input to router: h [B,S,D]

  - Flatten: h2 = h.view(B*S, D) → [T,D] where T=B*S
  - router_logits = linear(h2, gate.weight) where gate.weight is shaped [E_routed, D] but applied as [T,E_routed].
      - Output: [T, E_routed] where E_routed = n_routed_experts.

  ### 2.3 Routing nonlinearity + correction bias usage

  - r = sigmoid(router_logits) → [T, E_routed] (float32 sigmoid, then used for math)
  - r_choice = r + e_score_correction_bias where bias shape [E_routed] broadcasts → [T,E_routed]
  - Important: bias affects selection only; weights are gathered from r, not r_choice.

  ### 2.4 Group routing (must support even if config uses 1)

  Let n_group = config.n_group, experts per group = E_routed / n_group.

  - Compute group scores:
      - reshape: r_choice.view(T, n_group, E_routed/n_group)
      - per group take top‑2 values, sum them → group_scores [T, n_group]
  - Choose groups: group_idx = topk(group_scores, k=topk_group, sorted=False) → [T, topk_group]
  - Build mask over experts in selected groups → score_mask [T, E_routed]
  - Zero out non-selected-group experts for choice:
      - scores_for_choice = r_choice.masked_fill(~score_mask, 0.0) → [T,E_routed]
  - Select experts: topk_indices = topk(scores_for_choice, k=K, sorted=False) → [T,K]
  - Gather weights from r (not r_choice): topk_weights = r.gather(dim=1, index=topk_indices) → [T,K]

  ### 2.5 Normalize + post-scale placement (GLM‑specific)

  - If norm_topk_prob: topk_weights /= (sum(topk_weights)+1e-20) per token.
  - Then: topk_weights *= routed_scaling_factor (2.5) after renorm.

  This differs from nmoe’s route_scale (pre-sigmoid).

  ### 2.6 Expert compute (packed 3D weights)

  GLM stores experts as:

  - gate_up_proj: [E_local, 2*Dff, D] where Dff = moe_intermediate_size
  - down_proj: [E_local, D, Dff]

  For each expert e hit:

  - gate_up = linear(x, gate_up_proj[e]) → [tokens_e, 2*Dff]
  - split: (gate, up) each [tokens_e, Dff]
  - y = act(gate) * up then y = linear(y, down_proj[e]) → [tokens_e, D]
  - multiply by routing weight for that token/expert, then scatter-add back to [T,D]

  ### 2.7 Shared experts (GLM only)

  After routed expert output reshaped back to [B,S,D]:

  - out = out + shared_experts(residuals)
    where shared_experts is a dense SwiGLU MLP with intermediate size moe_intermediate_size * n_shared_experts.

  ———

  ## 3) MiniMax‑M2: Attention (GQA + “full-projection” QK norm + RoPE dim from rope init)

  Config highlights: D=3072, H=48, KV=8, Dh=128, rotary_dim=64, rope_theta=5e6, use_qk_norm=True.

  ### 3.1 Projections (no bias)

  Input x [B,S,D]:

  - q_lin = q_proj(x) no bias → [B,S,H*Dh]
  - k_lin = k_proj(x) no bias → [B,S,KV*Dh]
  - v_lin = v_proj(x) no bias → [B,S,KV*Dh]

  ### 3.2 QK norm placement (MiniMax‑specific)

  If use_qk_norm:

  - q_lin = RMSNorm(H*Dh)(q_lin)  weight shape [H*Dh]
  - k_lin = RMSNorm(KV*Dh)(k_lin) weight shape [KV*Dh]

  This happens before view into heads.

  ### 3.3 View + transpose + RoPE

  - q = q_lin.view(B,S,H,Dh).transpose(1,2) → [B,H,S,Dh]
  - k = k_lin.view(B,S,KV,Dh).transpose(1,2) → [B,KV,S,Dh]
  - v = v_lin.view(B,S,KV,Dh).transpose(1,2) → [B,KV,S,Dh]

  RoPE is applied exactly like GLM’s apply_rotary_pos_emb:

  - rotates Dr = cos.shape[-1] dims, pass-through remainder.
  - For M2, Dr should be 64 (matches config rotary_dim).

  Cos/sin contract again: [B,S,Dr] then unsqueeze on dim 1.

  ### 3.4 Attention compute + output proj

  Same GQA expansion and causal softmax; output projection:

  - o_proj no bias; output [B,S,D].

  ———

  ## 4) MiniMax‑M2: MoE (sigmoid routing + correction bias + renorm; no shared experts in the HF file)

  Config highlights: E=256, K=8, router_jitter_noise=0, router_aux_loss_coef=0.001.

  ### 4.1 Routing math (no group routing)

  Input h [B,S,D], flatten T=B*S:

  - router_logits = gate(h.view(T,D)) → [T,E]
  - w = sigmoid(router_logits.float()) → [T,E]
  - scores_for_choice = w + e_score_correction_bias → [T,E]
  - topk_index = topk(scores_for_choice, k=K, sorted=False) → [T,K]
  - topk_weights = w.gather(1, topk_index) → [T,K]
  - Always renorm: topk_weights /= sum(topk_weights) per token.

  No routed_scaling_factor in the HF file.

  ### 4.2 Expert weights layout (ModuleList form in HF file)

  HF file builds E separate SwiGLU MLP modules, each with:

  - w1: [Dff, D], w3: [Dff, D], w2: [D, Dff] where Dff = intermediate_size (1536 in config).

  For our implementation we likely want a packed expert weight layout (like GLM or like nmoe’s W1/W3/W2 tensors) but semantics are equivalent.

  ———

  ## 5) Expected HF state_dict key patterns (minimum required to load)

  These are the things your loader must be able to find and map (names may vary slightly by checkpoint packaging, but HF modeling suggests these patterns).

  ### 5.1 GLM‑4.7

  Per layer i:

  - Norms:
      - model.layers.{i}.input_layernorm.weight [D]
      - model.layers.{i}.post_attention_layernorm.weight [D]
  - Attention:
      - model.layers.{i}.self_attn.q_proj.weight [H*Dh, D]
      - model.layers.{i}.self_attn.q_proj.bias [H*Dh]  (attention_bias=True)
      - model.layers.{i}.self_attn.k_proj.weight [KV*Dh, D]
      - model.layers.{i}.self_attn.k_proj.bias [KV*Dh]
      - model.layers.{i}.self_attn.v_proj.weight [KV*Dh, D]
      - model.layers.{i}.self_attn.v_proj.bias [KV*Dh]
      - model.layers.{i}.self_attn.o_proj.weight [D, H*Dh]
      - If use_qk_norm:
          - model.layers.{i}.self_attn.q_norm.weight [Dh]
          - model.layers.{i}.self_attn.k_norm.weight [Dh]
  - MLP (dense layers only, i < first_k_dense_replace):
      - model.layers.{i}.mlp.gate_proj.weight [Dff_dense, D]
      - model.layers.{i}.mlp.up_proj.weight [Dff_dense, D]
      - model.layers.{i}.mlp.down_proj.weight [D, Dff_dense]
  - MoE (layers i >= first_k_dense_replace):
      - Router:
          - model.layers.{i}.mlp.gate.weight [E_routed, D] (parameter)
          - model.layers.{i}.mlp.gate.e_score_correction_bias [E_routed] (buffer; may or may not be present depending on save)
      - Experts packed:
          - model.layers.{i}.mlp.experts.gate_up_proj [E_local, 2*Dff_moe, D]
          - model.layers.{i}.mlp.experts.down_proj [E_local, D, Dff_moe]
      - Shared experts:
          - model.layers.{i}.mlp.shared_experts.gate_proj.weight [(Dff_moe*n_shared), D]
          - model.layers.{i}.mlp.shared_experts.up_proj.weight same
          - model.layers.{i}.mlp.shared_experts.down_proj.weight [D, (Dff_moe*n_shared)]

  Global:

  - model.embed_tokens.weight [vocab, D]
  - lm_head.weight if untied (tie_word_embeddings=False in config)

  ### 5.2 MiniMax‑M2

  Per layer i:

  - Norms:
      - model.layers.{i}.input_layernorm.weight [D]
      - model.layers.{i}.post_attention_layernorm.weight [D]
  - Attention:
      - model.layers.{i}.self_attn.q_proj.weight [H*Dh, D]
      - model.layers.{i}.self_attn.k_proj.weight [KV*Dh, D]
      - model.layers.{i}.self_attn.v_proj.weight [KV*Dh, D]
      - model.layers.{i}.self_attn.o_proj.weight [D, H*Dh]
      - If use_qk_norm (full-projection norms):
          - model.layers.{i}.self_attn.q_norm.weight [H*Dh]
          - model.layers.{i}.self_attn.k_norm.weight [KV*Dh]
  - MoE block:
      - model.layers.{i}.block_sparse_moe.gate.weight [E, D]
      - model.layers.{i}.block_sparse_moe.e_score_correction_bias [E] (buffer; again may or may not be saved)
      - Experts as ModuleList (if stored that way):
          - model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight [Dff, D]
          - model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight [Dff, D]
          - model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight [D, Dff]

  Global:

  - model.embed_tokens.weight [vocab, D]
  - lm_head.weight if untied

  ———

  ## 6) Implementation invariants to assert early (fail-fast)

  - H % KV == 0 and Dh * H == q_proj.out_features.
  - GLM: if use_qk_norm, the q/k norm weights must be shaped [Dh] (per-head).
  - MiniMax: if use_qk_norm, q_norm is [H*Dh] and k_norm is [KV*Dh] (pre-view).
  - RoPE: cos/sin last dim must equal the rotary dim (GLM: int(Dh*partial_rotary_factor), M2: should match rotary_dim from config/rope init).
  - Router selection uses sigmoid(router_logits) and correction bias only for selection (weights gathered from the uncorrected sigmoid outputs).
  - GLM: apply norm_topk_prob before multiplying by routed_scaling_factor.

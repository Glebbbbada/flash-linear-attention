import torch
import triton
import triton.language as tl

from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
)

try:
    from fla.ops.utils.index import prepare_chunk_indices
except ImportError:
    prepare_chunk_indices = None

try:
    from fla.ops.quasar.forward_substitution import forward_substitution_kernel
except ImportError:
    forward_substitution_kernel = None


@triton.jit
def precompute_alpha_kernel(
    k_ptr, beta_ptr, alpha_ptr,
    T,
    H:  tl.constexpr,
    S:  tl.constexpr,
    BT: tl.constexpr,
    stride_kb, stride_kt, stride_kh, stride_ks,
    stride_ab, stride_at, stride_ah,
):
    pid = tl.program_id(0)
    NT_total = tl.cdiv(T, BT)
    i_c = pid % NT_total
    pid2 = pid // NT_total
    i_h = pid2 % H
    i_b = pid2 // H
    beta = tl.load(beta_ptr + i_b).to(tl.float32)
    r_s = tl.arange(0, S)
    r_bt = tl.arange(0, BT)
    t0 = i_c * BT
    r_t = t0 + r_bt
    mask_t = r_t < T
    base_k = k_ptr + i_b * stride_kb + i_h * stride_kh
    kT = tl.load(base_k + r_s[:, None] * stride_ks + r_t[None, :] * stride_kt,
                 mask=mask_t[None, :], other=0.0).to(tl.float32)
    k_sq = tl.sum(kT * kT, axis=0)
    alpha = (1.0 - tl.exp(-beta * k_sq)) / (k_sq + 1e-8)
    base_a = alpha_ptr + i_b * stride_ab + i_h * stride_ah
    tl.store(base_a + r_t * stride_at, alpha, mask=mask_t)


@triton.jit
def chunk_state_tiled_kernel(
    k_ptr, v_ptr, alpha_ptr, states_ptr,
    init_ptr,
    T,
    H:  tl.constexpr,
    S:  tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    NT,
    HAS_INIT: tl.constexpr,
    stride_kb, stride_kt, stride_kh, stride_ks,
    stride_vb, stride_vt, stride_vh, stride_vs,
    stride_ab, stride_at, stride_ah,
    stride_sb, stride_sh, stride_sn, stride_si, stride_sj,
):
    NS = S // BS
    pid = tl.program_id(0)
    i_s = pid % NS
    pid2 = pid // NS
    i_h = pid2 % H
    i_b = pid2 // H
    r_s_row = i_s * BS + tl.arange(0, BS)
    r_s_col = tl.arange(0, S)
    r_bt = tl.arange(0, BT)
    acc = tl.zeros((BS, S), dtype=tl.float32)
    if HAS_INIT:
        init_base = init_ptr + i_b * S * S * H + i_h * S * S
        acc = tl.load(init_base + r_s_row[:, None] * S + r_s_col[None, :]).to(tl.float32)
    base_k = k_ptr + i_b * stride_kb + i_h * stride_kh
    base_v = v_ptr + i_b * stride_vb + i_h * stride_vh
    base_a = alpha_ptr + i_b * stride_ab + i_h * stride_ah
    base_s = states_ptr + i_b * stride_sb + i_h * stride_sh
    for c in range(NT):
        tl.store(base_s + c * stride_sn + r_s_row[:, None] * stride_si + r_s_col[None, :] * stride_sj, acc)
        t0 = c * BT
        r_t = t0 + r_bt
        mask_t = r_t < T
        kT_tile = tl.load(base_k + r_s_row[:, None] * stride_ks + r_t[None, :] * stride_kt,
                          mask=mask_t[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(base_v + r_t[:, None] * stride_vt + r_s_col[None, :] * stride_vs,
                         mask=mask_t[:, None], other=0.0).to(tl.float32)
        alpha = tl.load(base_a + r_t * stride_at, mask=mask_t, other=0.0)
        kT_a = kT_tile * alpha[None, :]
        acc += tl.dot(kT_a.to(tl.bfloat16), v_tile.to(tl.bfloat16)).to(tl.float32)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['S', 'BT'],
)
@triton.jit
def chunk_output_fwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, states_ptr, o_ptr,
    T,
    H:  tl.constexpr,
    S:  tl.constexpr,
    BT: tl.constexpr,
    stride_qb, stride_qt, stride_qh, stride_qs,
    stride_kb, stride_kt, stride_kh, stride_ks,
    stride_vb, stride_vt, stride_vh, stride_vs,
    stride_ob, stride_ot, stride_oh, stride_os,
    stride_sb, stride_sh, stride_sn, stride_si, stride_sj,
):
    i_c  = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b  = i_bh // H
    i_h  = i_bh % H
    beta = tl.load(beta_ptr + i_b).to(tl.float32)
    t0   = i_c * BT
    r_t  = t0 + tl.arange(0, BT)
    r_s  = tl.arange(0, S)
    mask = r_t < T
    base_q = q_ptr + i_b * stride_qb + i_h * stride_qh
    base_k = k_ptr + i_b * stride_kb + i_h * stride_kh
    base_v = v_ptr + i_b * stride_vb + i_h * stride_vh
    q = tl.load(base_q + r_t[:, None] * stride_qt + r_s[None, :] * stride_qs, mask=mask[:, None], other=0.0).to(tl.bfloat16)
    kT = tl.load(base_k + r_s[:, None] * stride_ks + r_t[None, :] * stride_kt, mask=mask[None, :], other=0.0).to(tl.float32)
    v = tl.load(base_v + r_t[:, None] * stride_vt + r_s[None, :] * stride_vs, mask=mask[:, None], other=0.0).to(tl.bfloat16)
    base_s = states_ptr + i_b * stride_sb + i_h * stride_sh + i_c * stride_sn
    st = tl.load(base_s + r_s[:, None] * stride_si + r_s[None, :] * stride_sj).to(tl.bfloat16)
    o_acc = tl.dot(q, st).to(tl.float32)
    k_sq = tl.sum(kT * kT, axis=0)
    alpha = (1.0 - tl.exp(-beta * k_sq)) / (k_sq + 1e-8)
    kT_a = (kT * alpha[None, :]).to(tl.bfloat16)
    A = tl.dot(q, kT_a).to(tl.float32)
    row = tl.arange(0, BT)
    col = tl.arange(0, BT)
    A = tl.where((row[:, None] >= col[None, :]) & mask[:, None] & mask[None, :], A, 0.0)
    o_acc += tl.dot(A.to(tl.bfloat16), v).to(tl.float32)
    base_o = o_ptr + i_b * stride_ob + i_h * stride_oh
    tl.store(base_o + r_t[:, None] * stride_ot + r_s[None, :] * stride_os, o_acc.to(tl.bfloat16), mask=mask[:, None])


@input_guard
def chunk_quasar_fwd(q, k, v, beta, initial_state=None, output_final_state=False, cu_seqlens=None, chunk_indices=None, chunk_size=128, **kwargs):
    B, T, H, S = q.shape
    BT = chunk_size
    BS = 16
    NT = triton.cdiv(T, BT)
    NS = S // BS
    q = q.to(torch.bfloat16).contiguous()
    k = k.to(torch.bfloat16).contiguous()
    v = v.to(torch.bfloat16).contiguous()
    if beta.dim() > 1:
        beta = beta.view(B, -1).mean(dim=1)
    beta = beta.float().contiguous()
    o = torch.empty(B, T, H, S, dtype=torch.bfloat16, device=q.device)
    states = torch.zeros(B, H, NT, S, S, dtype=torch.float32, device=q.device)
    alpha_buf = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    has_init = initial_state is not None
    init_ptr = initial_state if has_init else torch.empty(0, device=q.device)

    # Phase 1: Precompute alpha (fully parallel, cheap)
    precompute_alpha_kernel[(B * H * NT,)](
        k, beta, alpha_buf,
        T, H=H, S=S, BT=BT,
        stride_kb=k.stride(0), stride_kt=k.stride(1), stride_kh=k.stride(2), stride_ks=k.stride(3),
        stride_ab=alpha_buf.stride(0), stride_at=alpha_buf.stride(1), stride_ah=alpha_buf.stride(2),
        num_warps=4, num_stages=3,
    )

    # Phase 2: Tiled state computation (4x more parallelism)
    chunk_state_tiled_kernel[(B * H * NS,)](
        k, v, alpha_buf, states, init_ptr,
        T, H=H, S=S, BS=BS, BT=BT, NT=NT, HAS_INIT=has_init,
        stride_kb=k.stride(0), stride_kt=k.stride(1), stride_kh=k.stride(2), stride_ks=k.stride(3),
        stride_vb=v.stride(0), stride_vt=v.stride(1), stride_vh=v.stride(2), stride_vs=v.stride(3),
        stride_ab=alpha_buf.stride(0), stride_at=alpha_buf.stride(1), stride_ah=alpha_buf.stride(2),
        stride_sb=states.stride(0), stride_sh=states.stride(1), stride_sn=states.stride(2), stride_si=states.stride(3), stride_sj=states.stride(4),
        num_warps=4, num_stages=3,
    )

    # Phase 3: Output kernel (unchanged)
    chunk_output_fwd_kernel[(NT, B * H)](
        q, k, v, beta, states, o,
        T, H=H, S=S, BT=BT,
        stride_qb=q.stride(0), stride_qt=q.stride(1), stride_qh=q.stride(2), stride_qs=q.stride(3),
        stride_kb=k.stride(0), stride_kt=k.stride(1), stride_kh=k.stride(2), stride_ks=k.stride(3),
        stride_vb=v.stride(0), stride_vt=v.stride(1), stride_vh=v.stride(2), stride_vs=v.stride(3),
        stride_ob=o.stride(0), stride_ot=o.stride(1), stride_oh=o.stride(2), stride_os=o.stride(3),
        stride_sb=states.stride(0), stride_sh=states.stride(1), stride_sn=states.stride(2), stride_si=states.stride(3), stride_sj=states.stride(4),
    )
    final_state = None
    if output_final_state:
        final_state = states[:, :, -1].to(q.dtype)
    return o, final_state


class ChunkQuasarFunction(torch.autograd.Function):
    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, initial_state, output_final_state, cu_seqlens):
        o, final_state = chunk_quasar_fwd(q, k, v, beta, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens)
        return o, final_state

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, d_final=None):
        return (None,) * 7


@input_guard
def chunk_quasar(q, k, v, beta, initial_state=None, output_final_state=False, cu_seqlens=None, chunk_size=128, **kwargs):
    return ChunkQuasarFunction.apply(q, k, v, beta, initial_state, output_final_state, cu_seqlens)

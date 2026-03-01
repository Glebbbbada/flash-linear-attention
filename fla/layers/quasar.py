from __future__ import annotations
import math
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.quasar import chunk_quasar, fused_recurrent_quasar
from fla.ops.quasar.gate import fused_quasar_gate
if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

class QuasarAttention(nn.Module):
    def __init__(self, hidden_size=2048, head_dim=128, num_heads=16, mode="chunk",
                 use_short_conv=True, conv_size=4, conv_bias=False, layer_idx=None,
                 norm_eps=1e-5, **kwargs):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.key_dim = int(num_heads * head_dim)
        self.value_dim = int(num_heads * head_dim)
        self.layer_idx = layer_idx
        assert mode in ["chunk", "fused_recurrent"]
        self.qkv_proj = nn.Linear(hidden_size, self.key_dim * 3, bias=False)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation="silu")
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation="silu")
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, bias=conv_bias, activation="silu")
        self.beta_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.beta_log._no_weight_decay = True
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2
        batch_size, q_len, _ = hidden_states.shape
        mode = "chunk"
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(x=q, cache=conv_state_q, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(x=k, cache=conv_state_k, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(x=v, cache=conv_state_v, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_dim) for x in (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)
        beta = F.softplus(self.beta_log)
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_quasar(q=q, k=k, v=v, beta=beta, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_quasar(q=q, k=k, v=v, beta=beta, initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        if past_key_values is not None:
            past_key_values.update(recurrent_state=recurrent_state, conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None, layer_idx=self.layer_idx, offset=q_len)
        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values

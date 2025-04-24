from .linear_attention import LinearAttentionState, LolcatsLinearAttention
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from flash_attn import flash_attn_func

EPS = 1e-6

class State(nn.Module):
    def __init__(self, C: int, G: int, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        self.C, self.G = C, G
        self.K = self.V = self.FK = self.H = self.S = None  # will be tensors after first call
        self.tokens_seen = 0
        self.max_cache_size = 2 * self.C + self.G
        self._dtype, self._device = dtype, device
        
        #For ring buffer
        self.write_ptr = 0                  # start of the “current window”
        self.valid     = 0                  # how many tokens we’ve cached

    def _init_state(self, k_c, fk_c):
        B, C, H, D = k_c.shape
        _, _, _, F = fk_c.shape
        dtype, device = k_c.dtype, k_c.device
        self.K  = torch.zeros((B, self.max_cache_size, H, D), device=device, dtype=dtype)
        self.V  = torch.zeros_like(self.K)
        self.FK = torch.zeros((B, self.max_cache_size, H, F), device=device, dtype=dtype)
        self.H  = torch.zeros((B, H, F, D), device=device, dtype=dtype)
        self.S  = torch.zeros((B, H, F),    device=device, dtype=dtype)

    def update(self, k_c, v_c, fk_c):
        B, C, H, D = k_c.shape
        if self.K is None:
            self._init_state(k_c, fk_c)

        # 1) if there is still room: simple write ----------------------------------
        if self.valid + C <= self.max_cache_size:
            pos = (self.write_ptr + self.valid) % self.max_cache_size
            self.K [:, pos:pos+C] = k_c          # one or two slices but copies
            self.V [:, pos:pos+C] = v_c
            self.FK[:, pos:pos+C] = fk_c
            self.valid += C
            return

        # 2) need to evict oldest tokens ------------------------------------------
        num_excess = self.valid + C - self.max_cache_size          # == tokens to evict

        # (a)  send the `num_excess` *oldest* items to linear-attn state
        old_fk = self._ring_view(self.FK, num_excess)
        old_v  = self._ring_view(self.V , num_excess)
        self._update_hidden_state(old_fk, old_v)

        # (b) advance write_ptr and keep cache full
        self.write_ptr = (self.write_ptr + num_excess) % self.max_cache_size
        self.valid = self.max_cache_size              # cache is now full

        # (c) write the new chunk at the *tail* (same math as the fast path)
        pos = (self.write_ptr + self.valid) % self.max_cache_size
        if pos + C <= self.max_cache_size:
            self.K [:, pos:pos+C] = k_c
            self.V [:, pos:pos+C] = v_c
            self.FK[:, pos:pos+C] = fk_c
        else:                                   # wrap once
            first = self.max_cache_size - pos
            self.K [:, pos:]      = k_c[:, :first]
            self.V [:, pos:]      = v_c[:, :first]
            self.FK[:, pos:]      = fk_c[:, :first]
            self.K [:, :C-first]  = k_c[:, first:]
            self.V [:, :C-first]  = v_c[:, first:]
            self.FK[:, :C-first]  = fk_c[:, first:]

    def _ring_view(self, tensor, length: int):
        """
        Return a view of the `length` left-most logical positions as a
        contiguous tensor, WITHOUT realloc / copy.  Shape is identical to
        `tensor[:, :length, …]` in the old code.
        """
        if self.write_ptr + length <= self.max_cache_size:
            return tensor[:, self.write_ptr:self.write_ptr+length]
        # wrap-around: return a cat-view, but this is tiny (<= 3 chunks)
        first = tensor[:, self.write_ptr:]              # tail
        second = tensor[:, :length - first.size(1)]     # head
        return torch.cat((first, second), dim=1)

    def linear_attn_forward(self, fq):
        """
        if self.H is None: return 0, EPS
        y_numerator = torch.einsum('b c h f, b h f d -> b c h d', fq, self.H)
        y_denominator = torch.einsum('b c h f, b h f -> b c h', fq, self.S).unsqueeze(-1)
        return y_numerator, y_denominator
        """
        B, C, H, F = fq.shape
        _, _, _, D = self.H.shape

        if self.H is None:
            return torch.zeros_like(fq[..., 0:1]), torch.full_like(fq[..., 0:1], EPS)

        #y_num = torch.einsum('b c h f, b h f d -> b c h d', fq, self.H) 
        y_num = torch.bmm(fq.transpose(1,2).flatten(0,1),  self.H.flatten(0,1)).view(B, C, H, D) #optimized einsu
        y_den = torch.einsum('b c h f, b h f -> b c h', fq, self.S).unsqueeze(-1)
        return y_num, y_den

    def _update_hidden_state(self, fk, v):
        self.H += torch.einsum('b c h f, b c h d -> b h f d', fk, v)
        self.S += fk.sum(1) # [B,H,F]
        
def _lola_forward(
        q, k, v, fq, fk, *, #[B, H, L, D]
        C: int,
        state: State,
        gate: torch.Tensor) -> Tuple[torch.Tensor, State]:
    """
    All tensors come in as [B,H,L,D] (or [B,H,L,F] for fq/fk).
    """
    #Pad the sequences, so it can be chunked.
    B, H, L, D = q.shape
    pad = (C - L % C) % C
    if pad:
        pad_qd = torch.zeros(B, H, pad, D,         dtype=q.dtype,  device=q.device)
        pad_f  = torch.zeros(B, H, pad, fq.size(-1), dtype=fq.dtype, device=q.device)
        q, k, v = [torch.cat([t, pad_qd], 2) for t in (q, k, v)]
        fq, fk  = [torch.cat([t, pad_f ], 2) for t in (fq, fk)]
        L += pad
    N = L // C

    q_c, k_c, v_c = (rearrange(t, 'b h (n c) d -> n b c h d', c=C) for t in (q, k, v))
    fq_c, fk_c =   (rearrange(t, 'b h (n c) f -> n b c h f', c=C) for t in (fq, fk))
    
    gate = gate.transpose(1,2).to(q.device, dtype=q.dtype) #(1,H,1,1)->(1,1,H,1)
    scaled_gate = gate / (1-gate)
    out = torch.empty((B, L, H, D), device=q.device, dtype=q.dtype)

    for n in range(N):
        #Update cache
        state.update(k_c[n], v_c[n], fk_c[n])

        #Sparse Attention
        y_soft, softmax_lse, _ = flash_attn_func(q_c[n], state._ring_view(state.K, state.valid), 
                                                state._ring_view(state.V, state.valid),
                                                causal=True, return_attn_probs=True)
        y_soft_den = torch.exp(softmax_lse).transpose(1,2).unsqueeze(-1)
        y_soft_num = y_soft * y_soft_den #(B C H D)

        #Linear Attention
        y_lin_num, y_lin_den = state.linear_attn_forward(fq_c[n])

        #Unoptimized Version:
        #y = (y_soft_num*(gate) + y_lin_num*(1-gate)) / (y_soft_den*(gate) + y_lin_den*(1-gate) + EPS)
        #Optimized Version:
        num = torch.addcmul(y_soft_num, scaled_gate, y_lin_num)
        den = torch.addcmul(y_soft_den, scaled_gate, y_lin_den).add_(EPS)
        y   = num / den
        
        out[:, n*C:(n+1)*C] = y


    return out[:, :L-pad], state

def lola_sparse_compatible(
    q, k, fq, fk, v, # [B,H,L,D]
    window_factor,
    *, window_size: int,
    global_cache_size: int,
    kv_state: Optional['LinearAttentionSparseSlidingWindowCache'] = None,
    **_,
):  
    if q.size(2) > 1:
        state = State(window_size, global_cache_size, q.dtype, q.device)
    else:
        state = kv_state.state if kv_state and kv_state.state else State(window_size, global_cache_size, q.dtype, q.device)

    if q.size(2) > 1: # pre‑fill / training
        y, state = _lola_forward(q, k, v, fq, fk, C=window_size, state=state, gate=window_factor)
    else: # generation
        raise 'Decode Not Implemented...'

    if kv_state is None:
        kv_state = LinearAttentionSparseSlidingWindowCache(
            window_size=window_size, global_cache_size=global_cache_size,
            dtype=q.dtype, device=q.device)
    kv_state.state = state
    return y, None, kv_state

class LolcatsSparseSlidingWindowAttention(LolcatsLinearAttention):
    def __init__(self, *, window_size=512, global_cache_size=512,
                 init_window_factor=0.0, train_window_factor=True, **kw):
        super().__init__(**kw)
        self.window_size = window_size
        self.global_cache_size = global_cache_size
        gate_init = init_window_factor * torch.ones(
            1, self.num_heads, 1, 1,
            dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device)
        if train_window_factor:
            self.window_factors = nn.Parameter(gate_init)
        else:
            self.register_buffer('window_factors', gate_init)
        self.quadratic_attention = lola_sparse_compatible

    def forward(self, hidden_states, attention_mask=None,
                position_ids=None, past_key_value=None,
                use_cache=False, **kw):
        q, k, v, _ = self.process_qkv(hidden_states, attention_mask,
                                      position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)
        y, _, new_state = self.quadratic_attention(
            q, k, f_q, f_k, v,
            torch.sigmoid(self.window_factors),
            window_size=self.window_size,
            global_cache_size=self.global_cache_size,
            kv_state=past_key_value)
        y = rearrange(y, 'b l h d -> b l (h d)')

        if use_cache:
            return self.o_proj(y), None, new_state
        return self.o_proj(y), None, None

class LinearAttentionSparseSlidingWindowCache(LinearAttentionState):
    def __init__(self, *, window_size=512, global_cache_size=512,
                 dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.state = State(window_size, global_cache_size, dtype, device)

# ---------------------------------------------------------------------------
#  linear_window_attention_sw_sparse.py  ―  LoLA (window + sparse + linear)
# ---------------------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_func
from typing import Optional, Callable

from .linear_attention import LinearAttentionState, LolcatsLinearAttention

# ---------------------------------------------------------------------------
# 1. Fixed‑RAM per‑layer state ---------------------------------------------
# ---------------------------------------------------------------------------
class LoLAState(nn.Module):
    """
    Sliding‑window + top‑G heap + low‑rank sums, constant size per layer.
    """
    def __init__(self, C:int, G:int,
                 dtype:torch.dtype, device:torch.device):
        super().__init__()
        self.C, self.G = C, G
        for n, dt in [("K_win",dtype),("V_win",dtype),
                      ("K_top",dtype),("V_top",dtype),("FK_top",dtype),
                      ("H_sum",torch.float32),("S_sum",torch.float32),
                      ("heap_val",torch.float32),("heap_idx",torch.long)]:
            self.register_buffer(n, torch.empty(0, device=device, dtype=dt))
        self.tokens_seen = 0                        # processed so far

    # ------------------------------------------------------------------ #
    def _init_from_first_chunk(self, k_c, v_c, fk_c, score_c):
        """Bootstrap heap & caches from first chunk."""
        B,H,C,D = k_c.shape
        G = min(self.G, C)
        top_val, top_idx = score_c.topk(G, -1)      # [B,H,G]

        self.heap_val, self.heap_idx = top_val, top_idx
        b,h = torch.arange(B)[:,None,None], torch.arange(H)[None,:,None]
        self.K_top  = k_c[b,h,top_idx]
        self.V_top  = v_c[b,h,top_idx]
        self.FK_top = fk_c[b,h,top_idx]
        self._recompute_sums()

    @torch.no_grad()
    def train_chunk(self, k_c, v_c, fk_c, score_c):
        B,H,C,D = k_c.shape
        dev     = k_c.device

        # 1) sliding window update (unchanged)
        if self.K_win.numel()==0:
            self.K_win, self.V_win = k_c.clone(), v_c.clone()
        else:
            self.K_win = torch.cat([self.K_win[:,:,1:], k_c[:,:,-1:]], 2)
            self.V_win = torch.cat([self.V_win[:,:,1:], v_c[:,:,-1:]], 2)

        # 2) build new heap
        chunk_idx = (self.tokens_seen + torch.arange(C, device=dev))\
                        .view(1,1,C).expand(B,H,C)           # [B,H,C]
        if self.heap_val.numel()==0:
            # bootstrap exactly as before
            self._init_from_first_chunk(k_c, v_c, fk_c, score_c)
        else:
            # a) scores and absolute indices
            cat_val = torch.cat([self.heap_val, score_c], dim=-1)     # [B,H,G+C]
            cat_idx = torch.cat([self.heap_idx, chunk_idx], dim=-1)   # [B,H,G+C]
            k_eff   = min(self.G, cat_val.size(-1))
            new_val, sel = cat_val.topk(k_eff, dim=-1)                # sel ∈ [0…G+C)
            new_idx      = torch.gather(cat_idx, -1, sel)             # absolute

            # b) gather new heap‐keys from old K_top + new chunk
            prefix_K  = torch.cat([self.K_top,  k_c], 2)  # [B,H,G+C,D]
            prefix_V  = torch.cat([self.V_top,  v_c], 2)
            prefix_FK = torch.cat([self.FK_top, fk_c], 2)

            # use sel to pick along dim=2
            idx_k   = sel.unsqueeze(-1).expand(-1,-1,-1,D)
            idx_fk  = sel.unsqueeze(-1).expand(-1,-1,-1,prefix_FK.size(-1))
            self.K_top  = torch.gather(prefix_K,  2, idx_k)
            self.V_top  = torch.gather(prefix_V,  2, idx_k)
            self.FK_top = torch.gather(prefix_FK, 2, idx_fk)

            # c) overwrite heap_val & heap_idx, recompute sums
            self.heap_val, self.heap_idx = new_val, new_idx
            self._recompute_sums()

        self.tokens_seen += C


    # ------------------------------------------------------------------ #
    def _recompute_sums(self):
        if self.K_top.numel()==0:
            self.H_sum = self.K_top
            self.S_sum = self.K_top
            return
        self.H_sum = torch.einsum('b h g f, b h g d -> b h f d',
                                  self.FK_top.float(), self.V_top.float())
        self.S_sum = self.FK_top.float().sum(2)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_token(self, k_t, v_t, fk_t):
        if self.K_win.numel()==0:
            self.K_win, self.V_win = k_t.unsqueeze(2), v_t.unsqueeze(2)
        else:
            self.K_win = torch.cat([self.K_win[:,:,1:], k_t.unsqueeze(2)], 2)
            self.V_win = torch.cat([self.V_win[:,:,1:], v_t.unsqueeze(2)], 2)

        self.H_sum += torch.einsum('b h f, b h d -> b h f d',
                                   fk_t.float(), v_t.float())
        self.S_sum += fk_t.float()
        self.tokens_seen += 1

# ---------------------------------------------------------------------------
# 2.  Fast forward (training / eval)  ---------------------------------------
# ---------------------------------------------------------------------------
def _lola_forward(q, k, v, fq, fk, *, C, G, state: LoLAState, alpha, beta):
    """
    All tensors shape [B, L, H, D]; alpha/beta broadcast [1,H,1,1].
    """
    B, L, H, D = q.shape
    pad = (C - L % C) % C
    if pad:
        pad_fn = lambda t: F.pad(t, (0,0,0,0,0,pad))
        q, k, v, fq, fk = map(pad_fn, (q, k, v, fq, fk))
    N = (L + pad) // C

    # === REPLACE view+transpose with rearrange ===
    q_c  = rearrange(q,  'b (n c) h d -> b h n c d', c=C)
    k_c  = rearrange(k,  'b (n c) h d -> b h n c d', c=C)
    v_c  = rearrange(v,  'b (n c) h d -> b h n c d', c=C)
    fq_c = rearrange(fq, 'b (n c) h f -> b h n c f', c=C)
    fk_c = rearrange(fk, 'b (n c) h f -> b h n c f', c=C)

    out = torch.empty((B, N*C, H, D), device=q.device, dtype=q.dtype)
    for n in range(N):
        qn, kn, vn = q_c[:,:,n], k_c[:,:,n], v_c[:,:,n]
        km1 = k_c[:,:,n-1] if n else kn[:,:,:0]
        vm1 = v_c[:,:,n-1] if n else vn[:,:,:0]

        K_blk = torch.cat([km1, kn, state.K_top], dim=2)
        V_blk = torch.cat([vm1, vn, state.V_top], dim=2)

        # softmax on window+top‑G
        yw = flash_attn_func(
            qn.flatten(0,1).unsqueeze(2),
            K_blk.flatten(0,1).unsqueeze(2),
            V_blk.flatten(0,1).unsqueeze(2),
            causal=False
        ).view(B, H, qn.size(2), D)

        # linear remainder
        y_lin = 0 if state.H_sum.numel()==0 else torch.einsum(
            'b h c f, b h f d -> b h c d',
            fq_c[:,:,n].float(), state.H_sum
        )

        yw    = alpha * yw
        y_lin = beta  * y_lin
        yy    = yw + y_lin

        # back to [B, n*C:(n+1)*C, H, D]
        out[:, n*C : n*C + yy.size(2)] = rearrange(yy, 'b h c d -> b c h d')

        # compute scores and update LoLAState
        resid   = (yw - y_lin).mean(2)
        score_c = torch.einsum('b h d, b h c d -> b h c',
                               resid.float(), vn.float()).abs()
        state.train_chunk(kn, vn, fk_c[:,:,n], score_c)

    return out[:, :L], state

# ---------------------------------------------------------------------------
# 3.  Generation (single token) --------------------------------------------
# ---------------------------------------------------------------------------
def _lola_decode(q,k,v,fq,fk, *, C,G, state:LoLAState,
                 alpha,beta):
    if state.tokens_seen==0:
        state.decode_token(k.squeeze(1), v.squeeze(1), fk.squeeze(1))
        return v, state

    B,H,D = q.shape[0], q.shape[2], q.shape[3]
    K_blk = torch.cat([state.K_win, state.K_top], 2)
    V_blk = torch.cat([state.V_win, state.V_top], 2)

    y_s = flash_attn_func(q.flatten(0,1).unsqueeze(2),
                          K_blk.flatten(0,1).unsqueeze(2),
                          V_blk.flatten(0,1).unsqueeze(2),
                          causal=False).view(B,H,1,D)

    y_l = 0 if state.H_sum.numel()==0 else torch.einsum(
            'b h 1 f, b h f d -> b h 1 d', fq.float(), state.H_sum)

    y = alpha*y_s + beta*y_l
    state.decode_token(k.squeeze(1), v.squeeze(1), fk.squeeze(1))
    return y, state

# ---------------------------------------------------------------------------
# 4.  Public wrapper --------------------------------------------------------
# ---------------------------------------------------------------------------
def lola_sparse_compatible(q,k,fq,fk,v,
        window_factor, linear_factor,
        *, window_size:int,
        kv_state:Optional['LinearAttentionSparseSlidingWindowCache']=None,
        **kw):

    C = window_size
    G = kv_state.global_cache_size if kv_state else 128
    alpha = window_factor               # shape [1,H,1,1]
    beta  = 1 - alpha                   # enforce convex gate

    st = kv_state.state if kv_state else None
    if q.size(2)>1:      # train / eval
        y, st = _lola_forward(q.permute(0,2,1,3), k.permute(0,2,1,3),
                              v.permute(0,2,1,3),
                              fq.permute(0,2,1,3), fk.permute(0,2,1,3),
                              C=C, G=G, state=st, alpha=alpha, beta=beta)
    else:                # generation
        y, st = _lola_decode(q.permute(0,2,1,3), k.permute(0,2,1,3),
                             v.permute(0,2,1,3),
                             fq.permute(0,2,1,3), fk.permute(0,2,1,3),
                             C=C, G=G, state=st, alpha=alpha, beta=beta)

    if kv_state is None:
        kv_state = LinearAttentionSparseSlidingWindowCache(
                        window_size=C, global_cache_size=G,
                        dtype=q.dtype, device=q.device)
    kv_state.state = st
    return y.permute(0,2,1,3), None, kv_state
# ---------------------------------------------------------------------------
# 5.  Attention module ------------------------------------------------------
# ---------------------------------------------------------------------------
class LolcatsSparseSlidingWindowAttention(LolcatsLinearAttention):
    def __init__(self, *, window_size=64, global_cache_size=128,
                 init_window_factor=0., train_window_factor=True, **kw):
        super().__init__(**kw)
        self.window_size = window_size
        self.global_cache_size = global_cache_size
        gate = init_window_factor*torch.ones(
                1, self.num_heads, 1, 1,
                device=self.q_proj.weight.device,
                dtype=self.q_proj.weight.dtype)
        if train_window_factor:
            self.window_factors = nn.Parameter(gate)
        else:
            self.register_buffer('window_factors', gate)
        self.quadratic_attention = lola_sparse_compatible

    def forward(self, hidden_states, attention_mask=None,
                position_ids=None, past_key_value=None,
                use_cache=False, **kw):

        # <<<  position_ids forwarded to avoid rotary‑emb bug  >>>
        q,k,v,_ = self.process_qkv(hidden_states, attention_mask,
                                   position_ids, past_key_value)

        f_q,f_k = self.feature_map_q(q), self.feature_map_k(k)
        y, _, new_state = self.quadratic_attention(
            q,k,f_q,f_k,v,
            torch.sigmoid(self.window_factors), 1,
            window_size=self.window_size,
            kv_state=past_key_value,
            global_cache_size=self.global_cache_size)

        y = rearrange(y, 'b h l d -> b l (h d)')
        return self.o_proj(y), None, (new_state if use_cache else None)
# ---------------------------------------------------------------------------
# 6.  Cache wrapper ---------------------------------------------------------
# ---------------------------------------------------------------------------
class LinearAttentionSparseSlidingWindowCache(LinearAttentionState):
    """
    Thin wrapper exposing LoLAState to the rest of the repo.
    """
    def __init__(self, *, window_size=64, global_cache_size=128,
                 dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.state = LoLAState(window_size, global_cache_size, dtype, device)
        self.kv_states.append(self.state.H_sum)
        self.k_states .append(self.state.S_sum)
        self.window_size, self.global_cache_size = window_size, global_cache_size

    # generation‑time hook
    def update_for_decoding(self, keys, values, *, feature_map_k, dtype, **kw):
        fk = feature_map_k(keys.to(dtype))
        self.state.decode_token(keys.squeeze(2), values.squeeze(2), fk.squeeze(2))
        self.kv_states[0], self.k_states[0] = self.state.H_sum, self.state.S_sum
        return self.state.K_win, self.state.V_win, self.state.H_sum, self.state.S_sum

    def reset(self):
        self.state.reset()
        self.kv_states[0], self.k_states[0] = self.state.H_sum, self.state.S_sum

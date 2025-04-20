# ---------------------------------------------------------------------------
# linear_window_attention_sw_sparse.py ― LoLA (window + sparse + linear)
# Pure‑PyTorch (no Triton) — causal, naturally‑mixed, with shape asserts
# ---------------------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from .linear_attention import LinearAttentionState, LolcatsLinearAttention

DEBUG_SHAPES = False       # flip to True for verbose shape prints
EPS          = 1e-6           # numerical stabiliser
# -------------------------------------------------------------------------


def _dbg(tag: str, *tensors):
    if DEBUG_SHAPES:
        shapes = ", ".join(str(tuple(t.shape)) for t in tensors)
        print(f"[DBG] {tag:<16}: {shapes}")


def _apply_gate(y_nat: torch.Tensor,
                y_soft: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
    """
    y_*   : [B,H,C,D]  or [B,H,1,D]
    gate  : broadcastable [1,H₀,1,1]   (H can be H₀ or a multiple of it)
    """
    if gate.size(1) != y_nat.size(1):
        assert y_nat.size(1) % gate.size(1) == 0, \
            f"gate heads {gate.size(1)} !| y heads {y_nat.size(1)}"
        gate = gate.repeat_interleave(y_nat.size(1) // gate.size(1), dim=1)
    return (1. - gate) * y_nat + gate * y_soft


# ==========================================================================
# 1.  Per‑layer state  (unchanged except variable‑length FK_top / V_top) ===
# ==========================================================================
class LoLAState(nn.Module):
    """
    LoLA state:
        sliding window  (K_win / V_win, length C)
        top-G KV heap   (K_top / V_top, length G)
        low rank linear attn   (H_sum / S_sum)  for *all remaining* tokens
    """
    def __init__(self, C: int, G: int,
                 dtype: torch.dtype,
                 device: torch.device | str):
        super().__init__()
        self.C, self.G = C, G
        for name, dt in [
            ("K_win", dtype), ("V_win", dtype), ("FK_win", dtype), ("win_score", torch.float32),
            ("K_top", dtype), ("V_top", dtype), ("FK_top", dtype),
            ("H_sum", torch.float32), ("S_sum", torch.float32),
            ("heap_score", torch.float32), ("heap_idx", torch.long),
        ]:
            self.register_buffer(name, torch.empty(0, device=device, dtype=dt))
        self.tokens_seen = 0


    # -------------------------------------------------------------- public
    @torch.no_grad()
    def train_chunk(self,
                    k_c: torch.Tensor,        # [B,H,C,D]
                    v_c: torch.Tensor,        # [B,H,C,D]
                    fk_c: torch.Tensor,       # [B,H,C,F]
                    score_c: torch.Tensor):   # [B,H,C]
        """Update sliding window + heap + low‑rank sums with one chunk"""
        B, H, C, D = k_c.shape
        _dbg("train_chunk/in", k_c, score_c)

        # ---- sliding window --------------------------------------------
        if self.K_win.numel() == 0:
            self.K_win, self.V_win, self.FK_win, self.win_score = k_c.clone(), v_c.clone(), fk_c.clone(), score_c.clone()
        else:
            # ----heap update --------------------------------------------
            if self.heap_score.numel() == 0:
                #INIT HEAP
                if self.G >= C:
                    #Chunk fits in Heap
                    self.K_top = self.K_win
                    self.V_top = self.V_win
                    self.FK_top = self.FK_win
                    self.heap_score = self.win_score
                else:
                    #Store only top-G of chunk in heap
                    self.heap_score, top_idx  = self.win_score.topk(self.G, -1,largest=True) # [B,H,G]

                    b = torch.arange(B, device=k_c.device)[:, None, None]
                    h = torch.arange(H, device=k_c.device)[None, :, None]
                    self.K_top  = self.K_win[b, h, top_idx]
                    self.V_top  = self.V_win[b, h, top_idx]
                    self.FK_top  = self.FK_win[b, h, top_idx]

                    #Store the rest in hidden state
                    _, bot_idx = self.win_score.topk(C-self.G, -1,largest=False) # [B,H,C-G]
                    self.H_sum = torch.einsum('b h g f, b h g d -> b h f d',
                                            self.FK_win[b,h,bot_idx].float(), self.V_win[b,h,bot_idx].float())
                    self.S_sum = self.FK_win[b,h,bot_idx].float().sum(2) # [B,H,F]

                    assert not (top_idx.unsqueeze(-1) == bot_idx.unsqueeze(-2)).any().item(), "A KV PAIR CANT BE BOTH TOP AND BOTTOM" 

            elif self.heap_score.size(-1)+C <= self.G:
                #Heap not full
                self.K_top = torch.cat([self.K_top,self.K_win], 2)
                self.V_top =  torch.cat([self.V_top,self.V_win], 2)
                self.FK_top =  torch.cat([self.FK_top,self.FK_win], 2)
                self.heap_score =  torch.cat([self.heap_score,self.win_score], 2)
            
            else:
                #Compare old heap and chunk leaving sliding window
                cat_score = torch.cat([self.heap_score, self.win_score], 2)   # [B,H,G+C]
                cat_K  = torch.cat([self.K_top,self.K_win], 2)
                cat_V  = torch.cat([self.V_top,self.V_win], 2)
                cat_FK = torch.cat([self.FK_top,self.FK_win], 2)

                sorted_idx = cat_score.argsort(dim=-1, descending=True)            # [B,H,G + C]
                top_idx = sorted_idx[..., :self.G]                                 # [B,H,G]
                bot_idx = sorted_idx[..., self.G:]                                 # [B,H,C]
                self.heap_score = torch.gather(cat_score, dim=2, index=top_idx)    # [B,H,G]
                
                self.K_top = torch.gather(cat_K, dim=2, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,cat_K.size(-1)))
                self.V_top = torch.gather(cat_V, dim=2, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,cat_V.size(-1)))
                self.FK_top = torch.gather(cat_FK, dim=2, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,cat_FK.size(-1)))

                #Store the rest in hidden state
                bot_FK = torch.gather(cat_FK, dim=2, index=bot_idx.unsqueeze(-1).expand(-1,-1,-1,cat_FK.size(-1)))  # [B,H,C,F]
                bot_V  = torch.gather(cat_V, dim=2, index=bot_idx.unsqueeze(-1).expand(-1,-1,-1,cat_V.size(-1)))   # [B,H,C,D]

                self.H_sum = torch.einsum('b h g f, b h g d -> b h f d',
                                        bot_FK.float(), bot_V.float())
                self.S_sum = bot_FK.float().sum(2) # [B,H,F]


            self.K_win = k_c
            self.V_win = v_c
            self.FK_win = fk_c
            self.win_score = score_c

        self.tokens_seen += C
        _dbg("train_chunk/out K_top", self.K_top)

    # ~~~~~~~~~~~~ decode ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @torch.no_grad()
    def decode_token(self, k_t: torch.Tensor, v_t: torch.Tensor, fk_t: torch.Tensor):
        """Append a single KV ( [B,H,D] / [B,H,F] ) during generation"""
        
        #NOTE: WE ARE NOT USING CACHING IN DECODING STEP YET.
        if self.K_win.numel() == 0:
            self.K_win, self.V_win, self.FK_win = k_t.unsqueeze(2), v_t.unsqueeze(2), fk_t.unsqueeze(2)
        else:
            #Store the pair leaving the window in the Hidden State
            self.H_sum += torch.einsum('b h f, b h d -> b h f d',
                                   self.FK_win[:,:,:1].float(), self.V_win[:,:,:1].float())
            self.S_sum +=  self.FK_win[:,:,:1].float()
            
            #Slide the window
            self.K_win = torch.cat([self.K_win[:, :, 1:], k_t.unsqueeze(2)], 2)
            self.V_win = torch.cat([self.V_win[:, :, 1:], v_t.unsqueeze(2)], 2)
            self.FK_win = torch.cat([self.FK_win[:, :, 1:], fk_t.unsqueeze(2)], 2)

       
        self.tokens_seen += 1


# ==========================================================================
# 2.  Core: soft‑max on (window ∪ heap)  +  linear on “others” ============
# ==========================================================================
def _softmax_window_plus_heap(q, Kw, Vw, Kh, Vh):
    """
    Soft‑max over concatenation of
        – window keys (Kw,Vw)   length ≤ 2C
        – heap   keys (Kh,Vh)   length = G
    Shapes: q [B,H,C,D]   Kw/Vw [B,H,S,D]   Kh/Vh [B,H,G,D]
    Returns (y_soft, Z_soft) with y_soft in [B,H,C,D]  (unnormed numerator),
                              Z_soft  in [B,H,C,1]
    """
    if Kh.numel() == 0:
        K_all, V_all = Kw, Vw
    else:
        K_all = torch.cat([Kw, Kh], 2)
        V_all = torch.cat([Vw, Vh], 2)

    scale  = K_all.size(-1) ** -0.5
    logits = torch.einsum('b h q d, b h k d -> b h q k',
                          q.float(), K_all.float()) * scale

    S = Kw.size(2)
    G = 0 if Kh.numel() == 0 else Kh.size(2)
    C = q.size(2)
    len_prev = S - C                     # keys from previous chunk
    causal = torch.ones((q.size(2), K_all.size(2)), device=q.device).tril(len_prev).bool()[None,None,:]
    logits = logits.masked_fill(~causal, -1e9)

    #exp_l   = logits.exp()                          # [B,H,C,S+G]
    exp_l = torch.exp(logits - logits.amax(dim=-1, keepdim=True)) #NOTE: We should be using this.
    Z_soft  = exp_l.sum(-1, keepdim=True)
    y_soft  = torch.einsum('... q k, ... k d -> ... q d', exp_l, V_all.float())
    return y_soft, Z_soft


def _linear_other(q_f,       # [B,H,C,F]
                  fk_sum,    # [B,H,F,D]
                  s_sum):     # [B,H,F]
                        
    """
    Low‑rank path on *all* tokens that live outside window ∪ heap.
    """
    if fk_sum.numel() == 0:
        return 0, 0
    y_lin = torch.einsum('b h c f, b h f d -> b h c d', q_f.float(), fk_sum)
    Z_lin = torch.einsum('b h c f, b h f -> b h c',    q_f.float(), s_sum).unsqueeze(-1)
    return y_lin, Z_lin


# ==========================================================================
# 3.  Chunked forward ======================================================
# ==========================================================================
def _lola_forward(
        q, k, v, fq, fk, *,
        C: int,
        state: LoLAState,
        gate: torch.Tensor) -> Tuple[torch.Tensor, LoLAState]:
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

    # chunk views ----------------------------------------------------------
    q_c, k_c, v_c = (rearrange(t, 'b h (n c) d -> n b h c d', c=C)
                     for t in (q, k, v))
    fq_c, fk_c =   (rearrange(t, 'b h (n c) f -> n b h c f', c=C)
                     for t in (fq, fk))

    out = torch.empty_like(q)
    for n in range(N):
        qn, kn, vn  = q_c[n], k_c[n], v_c[n]
        fq_n, fk_n  = fq_c[n], fk_c[n]

        prev_k, prev_v = (k_c[n-1], v_c[n-1]) if n else (kn[:, :, :0], vn[:, :, :0])
        prev_fk        = fk_c[n-1]           if n else fk_n[:, :, :0]

        # ---- soft‑max over window ∪ heap --------------------------------
        Kw = torch.cat([prev_k, kn], 2)
        Vw = torch.cat([prev_v, vn], 2)
        y_soft_num, Z_soft = _softmax_window_plus_heap(
            qn, Kw, Vw, state.K_top, state.V_top)

        # ---- linear on "others" -----------------------------------------
        #   cumulative sums in state already exclude window & heap keys
        y_lin_num, Z_lin = _linear_other(fq_n, state.H_sum, state.S_sum)

        # ---- natural mixture & gate -------------------------------------
        #y_nat = (y_soft_num + y_lin_num) / (Z_soft + Z_lin + EPS)
        #y     = _apply_gate(y_nat, y_soft_num / (Z_soft + EPS), gate)
        """
        if isinstance(y_lin_num,torch.Tensor):
            print('gate value across 5 heads', gate[0,:5,0,0])
            print('normalized ysoft', (y_soft_num / (Z_soft + EPS))[0,0,:5,0])
            print('normalized ylin', (y_lin_num / (Z_lin + EPS))[0,0,:5,0])
            print('ysoft_num', (y_soft_num)[0,0,:5,0])
            print('ylin_num', (y_lin_num)[0,0,:5,0])
            print('gated_num', (y_soft_num*gate + y_lin_num*(1-gate))[0,0,:5,0])
            print('Z_soft across 5 tokens', Z_soft[0,0,:5])
            print('Z_lin across 5 tokens', Z_lin[0,0,:5])
        """
        if isinstance(y_lin_num,torch.Tensor):
            y = (y_soft_num*(1-gate) + y_lin_num*(gate)) / (Z_soft*(1-gate) + Z_lin*(gate) + EPS) 
            #y = (y_soft_num*gate + y_lin_num*(1-gate)) / (Z_soft*gate + Z_lin*(1-gate) + EPS) #NOTE: IDEAL GATING, BUT THE ONE ABOVE GETS 48%. THIS GETS 0%.
        else:
            y = y_soft_num*(1-gate) / (Z_soft*(1-gate) + EPS)

        out[:, :, n*C:(n+1)*C] = y

        # ---- residual score (error if we linearise the *window*) --------
        lin_win_num, lin_win_den = _linear_other(fq_n,
                                                 torch.einsum('b h k f, b h k d -> b h f d',
                                                              torch.cat([prev_fk, fk_n], 2).float(),
                                                              Vw.float()),
                                                 torch.cat([prev_fk, fk_n], 2).float().sum(2))
        y_lin_win = lin_win_num / (lin_win_den + EPS)
        resid     = (y_soft_num / (Z_soft + EPS) - y_lin_win).detach().mean(2)
        score_c   = torch.einsum('b h d, b h c d -> b h c', resid.float(), vn.float()).abs()

        state.train_chunk(kn, vn, fk_n, score_c)

    return out[:, :, :L-pad], state 


# ==========================================================================
# 4.  Decode (one token) ===================================================
# ==========================================================================

#NOTE: WE ARE NOT USING SPARSE CACHING FOR DECODING YET.
@torch.no_grad()
def _lola_decode(
        q, k, v, fq, fk, *,
        C: int,
        state: LoLAState,
        gate: torch.Tensor):
    print('DECODING: ERORR NOT SUPPORTED I THINK')
    if state.tokens_seen == 0:                         # bootstrap
        state.decode_token(k.squeeze(2), v.squeeze(2), fk.squeeze(2))
        return v, state

    y_soft_num, Z_soft = _softmax_window_plus_heap(
        q, state.K_win, state.V_win,
        state.K_top, state.V_top)

    y_lin_num, Z_lin = _linear_other(fq, state.H_sum, state.S_sum)
    y_nat  = (y_soft_num + y_lin_num) / (Z_soft + Z_lin + EPS)
    y_out  = _apply_gate(y_nat, y_soft_num / (Z_soft + EPS), gate)

    state.decode_token(k.squeeze(2), v.squeeze(2), fk.squeeze(2))
    return y_out, state


# ==========================================================================
# 5.  Public wrapper (permutes once) =======================================
# ==========================================================================
def lola_sparse_compatible(
    q, k, fq, fk, v,
    window_factor, linear_factor,      # linear_factor kept for API parity
    *, window_size: int,
    global_cache_size: int,
    kv_state: Optional['LinearAttentionSparseSlidingWindowCache'] = None,
    **_,
):
    C = window_size
    state = kv_state.state if kv_state else LoLAState(C, global_cache_size, q.dtype, q.device)
    gate  = window_factor.to(dtype=q.dtype)            # [1,H₀,1,1]

    # external [B,L,H,D] → internal [B,H,L,D]
    qh, kh, vh, fqh, fkh = (t for t in (q, k, v, fq, fk)) # was t.permute(0, 2, 1, 3)
    _dbg("wrapper/in", qh, gate)

    if q.size(1) > 1:                          # pre‑fill / training
        y_h, state = _lola_forward(qh, kh, vh, fqh, fkh,
                                   C=C, state=state, gate=gate)
    else:                                      # generation
        y_h, state = _lola_decode(qh, kh, vh, fqh, fkh,
                                  C=C, state=state, gate=gate)

    y = y_h.permute(0, 2, 1, 3)                # back to [B,L,H,D]

    if kv_state is None:
        kv_state = LinearAttentionSparseSlidingWindowCache(
            window_size=C, global_cache_size=global_cache_size,
            dtype=q.dtype, device=q.device)
    kv_state.state = state
    return y, None, kv_state


# ==========================================================================
# 6.  HF module wrapper ====================================================
# ==========================================================================
class LolcatsSparseSlidingWindowAttention(LolcatsLinearAttention):
    def __init__(self, *, window_size=64, global_cache_size=128,
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
            torch.sigmoid(self.window_factors), 1.0,
            window_size=self.window_size,
            global_cache_size=self.global_cache_size,
            kv_state=past_key_value)
        y = rearrange(y, 'b l h d -> b l (h d)')
        return self.o_proj(y), None, (new_state if use_cache else None)


# ==========================================================================
class LinearAttentionSparseSlidingWindowCache(LinearAttentionState):
    def __init__(self, *, window_size=64, global_cache_size=128,
                 dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.state = LoLAState(window_size, global_cache_size, dtype, device)
        self.kv_states.append(self.state.H_sum)
        self.k_states.append(self.state.S_sum)
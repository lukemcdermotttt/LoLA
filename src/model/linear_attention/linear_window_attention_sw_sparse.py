# ---------------------------------------------------------------------------
# linear_window_attention_sw_sparse.py ― LoLA (window + sparse + linear)
# Pure‑PyTorch (no Triton) — causal, naturally‑mixed, with shape asserts
# ---------------------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from .linear_attention import LinearAttentionState, LolcatsLinearAttention
from flash_attn import flash_attn_func


DEBUG_SHAPES = True      # flip to True for verbose shape prints
EPS          = 1e-6           # numerical stabiliser
# -------------------------------------------------------------------------


def _dbg(tag: str, *tensors):
    if DEBUG_SHAPES:
        shapes = ", ".join(str(tuple(t.shape)) for t in tensors)
        print(f"[DBG] {tag:<16}: {shapes}")



class State(nn.Module):
    def __init__(self, C: int, G: int, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        """
        Cache is set up, so K V and FK hold both sparse cache and sliding window (including current chunk)
        K = [ SPARSE CACHE | LAST CHUNK | CURRENT CHUNK ] #shape (B,max_cache_size,H,D) ... same for V, FK except FK has a final F dim instead of D
        
        """
        self.C, self.G = C, G
        self.K, self.V, self.FK, self.H, self.S = None, None, None, None, None #check _init_state for info
        self.tokens_seen = 0
        self.max_cache_size = 2*self.C + self.G
        

    def _init_state(self, k_c, fk_c):
        B, C, H, D = k_c.shape
        _, _, _, F = fk_c.shape
        dtype, device = k_c.dtype, k_c.device

        self.K = torch.zeros((B,self.max_cache_size,H,D),device=device,dtype=dtype) #Keys
        self.V = torch.zeros((B,self.max_cache_size,H,D),device=device,dtype=dtype) #Values
        self.FK = torch.zeros((B,self.max_cache_size,H,F),device=device,dtype=dtype) #Phi(K) for Linear Attn.
        self.H = torch.zeros((B,H,F,D),device=device,dtype=dtype) #Hidden State in Linear Attn.
        self.S = torch.zeros((B,H,F),device=device,dtype=dtype) #Normalizing State in Linear Attn.

    def update(self, k_c, v_c, fk_c):
        B, C, H, D = k_c.shape
        if self.K is None: self._init_state(k_c, fk_c)

        if self.tokens_seen + C < self.max_cache_size: #Cache isn't full
            self.K[:,self.tokens_seen: self.tokens_seen+C] = k_c
            self.V[:,self.tokens_seen: self.tokens_seen+C] = v_c
            self.FK[:,self.tokens_seen: self.tokens_seen+C] = fk_c
            self.tokens_seen += C
            #return 
        
        else: #Make space in the cache by sending excess tokens to the hidden state
            num_excess_tokens = self.G+self.C-self.tokens_seen-1 #TODO: Triple check this is correct.
            num_comparisons = self.G+num_excess_tokens #TODO: Make this whole thing simpler. Only using this rn as its explainable.

            #What is the current memory's prediction of the value? Keep the KV-pairs with high prediction error
            #Pretend the key is a future query
            numerator, denominator = self.linear_attn_forward(self.FK[:,:num_comparisons]) # (B, num_comparisons, H, D)
            V_pred = numerator / denominator
            pred_err = torch.norm(V_pred - self.V[:,:num_comparisons],dim=-1) # (B, num_comparisons, H)

            sorted_idx = pred_err.argsort(dim=1, descending=False) #NOTE: This is slow. Using this for now as its simple. (B, num_comparisons, H) 
            self.K[:,:num_comparisons] = self.K[:,:num_comparisons][sorted_idx] #Ensure the top-G errors are the last G-many in [0, num_comparisons]
            self.V[:,:num_comparisons] = self.V[:,:num_comparisons][sorted_idx]
            self.FK[:,:num_comparisons] = self.FK[:,:num_comparisons][sorted_idx]
            
            self._update_hidden_state(self.FK[:,:num_comparisons-self.G],self.V[:,:num_comparisons-self.G]) #Add low error pairs to self.H & self.S

            #roll num_comparisons-self.G to the left, ensuring K is [ Top G Pairs | Last Chunk | Low Error Pairs ]
            self.K = torch.roll(self.K, -(num_comparisons-self.G), 1)
            self.V = torch.roll(self.V, -(num_comparisons-self.G), 1)
            self.FK = torch.roll(self.FK, -(num_comparisons-self.G), 1)

            #Then replace the low error pairs with the incoming chunk since we already updated the hidden state.
            self.K[-(num_comparisons-self.G):] = k_c
            self.V[-(num_comparisons-self.G):] = v_c
            self.FK[-(num_comparisons-self.G):] = fk_c

            self.tokens_seen += C
            #return


            #make space. we have  self.max_cache_size-(self.tokens_seen+C+1) too many tokens
            #self.G + self.max_cache_size-(self.tokens_seen+C+1) --> self.G
            #G + (2C + G) - (tokens_seen + C + 1) = 2G + C - tokens_seen - 1 -> G
            #score the first [:(2G+C-self.tokens_seen-1)] & take top self.G
            

    def linear_attn_forward(self, fq):
        if self.H_sum is None: return 0, EPS
        y_numerator = torch.einsum('b c h f, b h f d -> b c h d', fq, self.H_sum)
        y_denominator = torch.einsum('b c h f, b h f -> b c h', fq, self.S_sum).unsqueeze(-1)
        return y_numerator, y_denominator

    def _update_hidden_state(self, fk, v):
        self.H_sum += torch.einsum('b c h f, b c h d -> b h f d', fk, v)
        self.S_sum += fk.sum(1) # [B,H,F]
        
def _new_lola_forward(
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
    out = torch.empty((B, L, H, D), device=q.device, dtype=q.dtype)

    for n in range(N):
        #Update cache
        state.update(k_c[n], v_c[n], fk_c[n])

        #Sparse Attention
        y_soft, softmax_lse, _ = flash_attn_func(q_c[n], state.K[:,:state.tokens_seen], 
                                                               state.V[:,:state.tokens_seen],
                                                            causal=True, return_attn_probs=True)
        y_soft_den = torch.exp(softmax_lse).transpose(1,2).unsqueeze(-1)
        y_soft_num = y_soft * y_soft_den #(B C H D)

        #Linear Attention
        y_lin_num, y_lin_den = state.linear_attn_forward(fq_c[n])

        #Fuse outputs
        y = (y_soft_num*(gate) + y_lin_num*(1-gate)) / (y_soft_den*(gate) + y_lin_den*(1-gate) + EPS)
        out[:, n*C:(n+1)*C] = y


    return out[:, :L-pad], state



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
        #TODO: Get rid of dtypes here, just put it in register buffer

        for name, dt in [
            ("K_win", dtype), ("V_win", dtype), ("FK_win", dtype),
            ("K_top", dtype), ("V_top", dtype), ("FK_top", dtype),
            ("H_sum", dtype), ("S_sum", dtype),
            ("heap_score", dtype)
        ]:
            self.register_buffer(name, torch.empty(0, device=device, dtype=dt))
        self.tokens_seen = 0

        #self.K, self.V, self.FK = None, None, None


    @torch.no_grad()
    def train_chunk(self,
                    k_c: torch.Tensor,        # [B,C,H,D]
                    v_c: torch.Tensor,        # [B,C,H,D]
                    fk_c: torch.Tensor,       # [B,C,H,F]
                    score_c: torch.Tensor):   # [B,C,H]
        """Update sliding window + heap + low-rank sums with one chunk"""
        B, C, H, D = k_c.shape
        _, _, _, F = fk_c.shape
        assert k_c.shape    == (B, C, H, D)
        assert fk_c.shape    == (B, C, H, F)
        #_dbg("train_chunk/in", k_c, score_c)
        #print('updating chunk, start: ', self.G, self.K_win.size(), self.V_win.size(), self.K_top.size(), self.V_top.size())

        #incoming score needs to talk about the current window.

        # ---- sliding window --------------------------------------------
        if self.K_win.numel() == 0:
            self.K_win, self.V_win, self.FK_win = k_c.clone(), v_c.clone(), fk_c.clone()
            self.H_sum = torch.zeros((B,H,F,D),device=k_c.device,dtype=k_c.dtype)
            self.S_sum = torch.zeros((B,H,F),device=k_c.device,dtype=k_c.dtype) 
        elif self.G == 0:
            #LoLCATs basically
            self.H_sum += torch.einsum('b c h f, b c h d -> b h f d', self.FK_win, self.V_win)
            self.S_sum += self.FK_win.sum(1) # [B,H,F]
            self.K_win = k_c
            self.V_win = v_c
            self.FK_win = fk_c
        else:
            # ----heap update --------------------------------------------
            if self.heap_score.numel() == 0:
                #INIT HEAP
                if self.G >= C:
                    #Chunk fits in Heap
                    self.K_top = self.K_win
                    self.V_top = self.V_win
                    self.FK_top = self.FK_win
                    self.heap_score = score_c #TODO: change to score_c
                else:
                    #Store only top-G of chunk in heap
                    sorted_idx = score_c.argsort(dim=1, descending=True)            # [B,C,H] #TODO: change to score_c
                    top_idx = sorted_idx[:,:self.G,:]                                 # [B,G,H]
                    bot_idx = sorted_idx[:,self.G:,:]                                 # [B,C-G,H]
                    self.heap_score = torch.gather(score_c, dim=1, index=top_idx)    # [B,G,H] #TODO: change to score_c
                    
                    self.K_top = torch.gather(self.K_win, dim=1, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,self.K_win.size(-1)))
                    self.V_top = torch.gather(self.V_win, dim=1, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,self.V_win.size(-1)))
                    self.FK_top = torch.gather(self.FK_win, dim=1, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,self.FK_win.size(-1)))

                    #Store the rest in hidden state
                    bot_FK = torch.gather(self.FK_win, dim=1, index=bot_idx.unsqueeze(-1).expand(-1,-1,-1,self.FK_win.size(-1)))  # [B,C,H,F]
                    bot_V  = torch.gather(self.V_win, dim=1, index=bot_idx.unsqueeze(-1).expand(-1,-1,-1,self.V_win.size(-1)))   # [B,C,H,D]
                    self.H_sum += torch.einsum('b c h f, b c h d -> b h f d', bot_FK, bot_V)
                    self.S_sum += bot_FK.sum(1) # [B,H,F]


            elif self.heap_score.size(1)+C <= self.G:
                #Heap not full
                self.K_top = torch.cat([self.K_top,self.K_win], 1)
                self.V_top =  torch.cat([self.V_top,self.V_win], 1)
                self.FK_top =  torch.cat([self.FK_top,self.FK_win], 1)
                self.heap_score =  torch.cat([self.heap_score,score_c], 1) #TODO: change to score_c
            
            else:
                #Compare old heap and chunk leaving sliding window

                #NOTE: Just added this. Re-compute heap scores for current H_sum.
                predicted_value_num = torch.einsum('b c h f, b h f d -> b c h d', self.FK_top, self.H_sum)
                Z_predicted_value = torch.einsum('b c h f, b h f -> b c h', self.FK_top, self.S_sum).unsqueeze(-1)
                predicted_value = predicted_value_num / (Z_predicted_value + EPS)
                self.heap_score = torch.norm(self.V_top - predicted_value, dim=-1) #NOTE: Just added this
                assert self.heap_score.size(1) <= self.G

                cat_score = torch.cat([self.heap_score, score_c], 1)   # [B,G+C,H] #TODO: change to score_c
                cat_K  = torch.cat([self.K_top,self.K_win], 1)
                cat_V  = torch.cat([self.V_top,self.V_win], 1)
                cat_FK = torch.cat([self.FK_top,self.FK_win], 1)

                sorted_idx = cat_score.argsort(dim=1, descending=True)            # [B,G + C,H]
                top_idx = sorted_idx[:, :self.G,:]                                 # [B,G,H]
                bot_idx = sorted_idx[:,self.G:,:]                                 # [B,C,H]
                self.heap_score = torch.gather(cat_score, dim=1, index=top_idx)    # [B,G,H]
                
                self.K_top = torch.gather(cat_K, dim=1, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,cat_K.size(-1)))
                self.V_top = torch.gather(cat_V, dim=1, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,cat_V.size(-1)))
                self.FK_top = torch.gather(cat_FK, dim=1, index=top_idx.unsqueeze(-1).expand(-1,-1,-1,cat_FK.size(-1)))

                #Store the rest in hidden state
                bot_FK = torch.gather(cat_FK, dim=1, index=bot_idx.unsqueeze(-1).expand(-1,-1,-1,cat_FK.size(-1)))  # [B,C,H,F]
                bot_V  = torch.gather(cat_V, dim=1, index=bot_idx.unsqueeze(-1).expand(-1,-1,-1,cat_V.size(-1)))   # [B,C,H,D]

                self.H_sum += torch.einsum('b c h f, b c h d -> b h f d', bot_FK, bot_V)
                self.S_sum += bot_FK.sum(1) # [B,H,F]


            self.K_win = k_c
            self.V_win = v_c
            self.FK_win = fk_c

        self.tokens_seen += C


#BACK UP SOFTMAX
def _softmax_window_plus_heap(q, K_all, V_all):

    """
    Shapes: q [B,C,H,D]   K_all [B,G,H,D]
    Returns (y_soft, Z_soft) with y_soft in [B,C,H,D]  (unnormed numerator),
                              Z_soft  in [B,C,H,1]
    """

    scale  = K_all.size(-1) ** -0.5
    logits = torch.einsum('b c h d, b g h d -> b h c g',
                          q, K_all) * scale

    C = q.size(1)
    len_prev = K_all.size(1) - C                     # keys from previous chunk
    causal = torch.ones((q.size(1), K_all.size(1)), device=q.device).tril(len_prev).bool()[None,None,:]
    logits = logits.masked_fill(~causal, -1e9)
    exp_l = torch.exp(logits - logits.amax(dim=-1, keepdim=True)) #NOTE: We should be using this.
    Z_soft  = exp_l.sum(-1).transpose(-1,-2).unsqueeze(-1) #[B C H 1]
    y_soft  = torch.einsum('b h c g, b g h d -> b c h d', exp_l, V_all)
    return y_soft, Z_soft




# ==========================================================================
# 3.  Chunked forward ======================================================
# ==========================================================================
def _lola_forward(
        q, k, v, fq, fk, *, #[B, H, L, D]
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

    q_c, k_c, v_c = (rearrange(t, 'b h (n c) d -> n b c h d', c=C) for t in (q, k, v))
    fq_c, fk_c =   (rearrange(t, 'b h (n c) f -> n b c h f', c=C) for t in (fq, fk))
    
    gate = gate.transpose(1,2).to(q.device, dtype=q.dtype) #(1,H,1,1)->(1,1,H,1)
    out = torch.empty((B, L, H, D), device=q.device, dtype=q.dtype)

    for n in range(N):
        # Step 1 ---- soft‑max over window ∪ heap --------------------------------
        union_K = torch.cat([state.K_top, state.K_win, k_c[n]], dim=1) #cat (B C H D) along C
        union_V = torch.cat([state.V_top, state.V_win, v_c[n]], dim=1)

        y_soft, softmax_lse, S_dmask = flash_attn_func(q_c[n], union_K, union_V, causal=True, return_attn_probs=True) #softmax_lse is shape [B H C]
        Z_soft = torch.exp(softmax_lse).transpose(1,2).unsqueeze(-1)# was torch.exp(softmax_lse).transpose(-1,-2).unsqueeze(-1) since flash attention returns logsumexp
        y_soft_num = y_soft * Z_soft #B C H D 

        #y_soft_num, Z_soft = _softmax_window_plus_heap(q_c[n], union_K, union_V)
        #y_soft = y_soft_num / (Z_soft + EPS)
        #print('flash y vs. y soft: ', flash_y_soft[0,:5,0,0], y_soft[0,:5,0,0])

        # Step 2 ---- scores --------------------------------
        if n>0:
            #Norm of the predicted value error, aka the associated singular value-ish
            #score_c = torch.norm(v_c[n-1] - torch.einsum('b c h f, b h f d -> b c h d',fk_c[n-1], state.H_sum), dim=-1)
            
            predicted_value_num = torch.einsum('b c h f, b h f d -> b c h d', fk_c[n-1], state.H_sum)
            Z_predicted_value = torch.einsum('b c h f, b h f -> b c h', fk_c[n-1], state.S_sum).unsqueeze(-1)
            predicted_value = predicted_value_num / (Z_predicted_value + EPS)
            score_c = torch.norm(v_c[n-1] - predicted_value, dim=-1) #NOTE: Just added this

            #union_FK = torch.cat([state.FK_top, fk_c[n-1], fk_c[n]], dim=1) #cat (B C H F) along C
            #fqk = torch.einsum('b c h f, b g h f -> b c h g', fq_c[n], union_FK)
            #causal_mask = torch.ones_like(fqk[0,:,0,:]).tril(union_FK.size(1) - C)[None,:,None,:]
            #fqk = fqk * causal_mask
            #y_lin_win= torch.einsum('b c h g, b g h d -> b c h d', fqk, union_V) / (fqk.sum(-1,keepdims=True) + EPS)
            #resid     = (y_soft - y_lin_win).detach().mean(1)
            #score_c   = torch.einsum('b h d, b c h d -> b c h', resid, v_c[n]) #used to have .abs(), what if we normalized resid & v's?
            #score_c = score_c / (torch.norm(v_c[n],dim=-1) #* torch.norm(resid,dim=-1).unsqueeze(1))
        else:
            score_c = None
        

        # Step 3 ---- linear attention on others --------------------------------
        if state.H_sum.numel() == 0:
            out[:, n*C:(n+1)*C] = y_soft
        else:
            y_lin_num = torch.einsum('b c h f, b h f d -> b c h d', fq_c[n], state.H_sum)
            Z_lin = torch.einsum('b c h f, b h f -> b c h',    fq_c[n], state.S_sum).unsqueeze(-1)
            y = (y_soft_num*(gate) + y_lin_num*(1-gate)) / (Z_soft*(gate) + Z_lin*(1-gate) + EPS)  #This is how it should work.
            #y = (y_soft_num*(1-gate) + y_lin_num*(gate)) / (Z_soft*(1-gate) + Z_lin*(gate) + EPS) #This is how it worked last night...
            out[:, n*C:(n+1)*C] = y

        state.train_chunk(k_c[n], v_c[n], fk_c[n], score_c) #Update LoLA State

    return out[:, :L-pad], state 


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

    print('DECODING: THIS IS NOT UPDATED YET')
    #BUG: DECODE DOESNT UPDATE SPARSE CACHING YET
    #print(torch.norm(state.H_sum))
    #(B H L D) -> (B L H D)
    #gate = gate.transpose(1,2)

    q, k, v, fq, fk, gate = (t.transpose(1,2) for t in (q, k, v, fq, fk, gate))
    
    keys = torch.cat([state.K_top, state.K_win, k], dim=1)
    values = torch.cat([state.V_top, state.V_win, v], dim=1)
    y_soft, softmax_lse, S_dmask = flash_attn_func(q, keys, values, return_attn_probs=True, causal=True) #softmax_lse is shape [B H C]
    #Z_soft = torch.exp(softmax_lse).transpose(1,2).unsqueeze(-1)# was torch.exp(softmax_lse).transpose(-1,-2).unsqueeze(-1) since flash attention returns logsumexp
    #y_soft_num = y_soft * Z_soft

    #y_lin_num = torch.einsum('b c h f, b h f d -> b c h d', fq, state.H_sum)
    #Z_lin = torch.einsum('b c h f, b h f -> b c h',    fq, state.S_sum).unsqueeze(-1)
    #y_out = (y_soft_num*(gate) + y_lin_num*(1-gate)) / (Z_soft*(gate) + Z_lin*(1-gate) + EPS)  #This is how it should work.
    #y_out = (y_soft_num*(1-gate) + y_lin_num*(gate)) / (Z_soft*(1-gate) + Z_lin*(gate) + EPS) #This is how it worked last night...
    

    #Update LoLA State
    #state.H_sum += torch.einsum('b h f, b h d -> b h f d', state.FK_win[:,0], state.V_win[:,0])
    #state.S_sum += state.FK_win[:,0]
    #state.FK_win = torch.cat([state.FK_win[:,1:],fk],dim=1)
    state.K_win = torch.cat([state.K_win[:,1:],k],dim=1)
    state.V_win = torch.cat([state.V_win[:,1:],v],dim=1)
    state.tokens_seen += 1
    
    return y_soft.to(q.dtype), state
    #return y_out.to(q.dtype), state


# ==========================================================================
# 5.  Public wrapper (permutes once) =======================================
# ==========================================================================
def lola_sparse_compatible(
    q, k, fq, fk, v, # [B,H,L,D]
    window_factor,
    *, window_size: int,
    global_cache_size: int,
    kv_state: Optional['LinearAttentionSparseSlidingWindowCache'] = None,
    **_,
):
    if q.size(2) > 1:
        state = LoLAState(window_size, global_cache_size, q.dtype, q.device)
    else:
        # only reuse the cache for true step‑by‑step generation (q.size(1)==1)
        state = kv_state.state if kv_state and kv_state.state else LoLAState(window_size, global_cache_size, q.dtype, q.device)


    if q.size(2) > 1: # pre‑fill / training
        y, state = _lola_forward(q, k, v, fq, fk, C=window_size, state=state, gate=window_factor)
    else: # generation
        y, state = _lola_decode(q, k, v, fq, fk, C=window_size, state=state, gate=window_factor)

    if kv_state is None:
        kv_state = LinearAttentionSparseSlidingWindowCache(
            window_size=window_size, global_cache_size=global_cache_size,
            dtype=q.dtype, device=q.device)
    kv_state.state = state
    return y, None, kv_state



# ==========================================================================
# 6.  HF module wrapper ====================================================
# ==========================================================================
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

        return self.o_proj(y), new_state, None #(new_state if use_cache else None)


# ==========================================================================
class LinearAttentionSparseSlidingWindowCache(LinearAttentionState):
    def __init__(self, *, window_size=512, global_cache_size=512,
                 dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.state =LoLAState(window_size, global_cache_size, dtype, device)
        self.kv_states.append(self.state.H_sum)
        self.k_states.append(self.state.S_sum)
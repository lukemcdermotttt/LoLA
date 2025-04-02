"""
Subquadratic attention combining sliding window and linear attentions
- Using "standard" sliding windows
- Didactically computes outputs with n^2 attention weights for now
- Copied + adapted from linear_window_attention_tk.py for single-file reference

For each layer: 
- We first compute (softmax) attention over sliding windows
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""
from typing import List, Tuple, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache

from .linear_attention import (
    LolcatsLinearAttention, LinearAttentionState, 
    softmax_attention
)




# ----------------------
# Sliding window helpers
# ----------------------
def get_masks(window_size: int, q_len: int, k_len: int, 
              device: torch.device) -> tuple[torch.Tensor]:
    """
    Return masks for softmax and linear attention terms
    -> 1 is include, 0 is ignore
    """
    kwargs = {'device': device, 'dtype': int}
    causal_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len)
    linear_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len - window_size)
    window_mask = causal_mask - linear_mask
    # Return softmax mask (window), linear attention mask
    # -> shapes broadcast over (b, h, q_len, k_len)
    return window_mask[None, None, ...], linear_mask[None, None, ...]


def hybrid_attention_quadratic(q: torch.Tensor, k: torch.Tensor, 
                               f_q: torch.Tensor, f_k: torch.Tensor,
                               v: torch.Tensor,
                               window_factor: torch.Tensor,
                               linear_factor: torch.Tensor,
                               window_size: int,
                               kv_state: torch.Tensor = None,
                               k_state: torch.Tensor = None,
                               eps: float = 1e-12,
                               mask_value: float = -1e8,
                               global_cache_size: int = 16,
                               return_global_cache: bool = False):
    
    """
    Hybrid attention that combines local sparse (sliding window) attention,
    global sparse attention, and low-rank (linear) attention.
    
    This version is refactored for memory efficiency using torch.matmul in place of einsum,
    while replicating the behavior of the original implementation exactly.
    """
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    scale = k.shape[-1] ** -0.5

    #print('q,k len: ', M,N)
    # Get the initial masks (assumed to be tensors of shape [B, H, M, N])
    mask_window, mask_linear = get_masks(window_size, M, N, q.device)

    #NOTE: PREFILL
    #The Errors with the first query token can be computed efficiently during prefill
    mask_window[:,:,0,:(M-N)] = 1 #(1,1,q_len,k_len) -> prefill size is k_len - q_len = M-N, 
    mask_linear[:,:,0,:(M-N)] = 0


    # Pre-cast inputs to float once.
    q_f   = q.float()
    k_f   = k.float()
    f_q_f = f_q.float()
    f_k_f = f_k.float()

    # ---------------------------
    # Sparse mask for global attention
    # ---------------------------
    # Compute low-rank attention scores.
    a_ln = torch.matmul(f_q_f, f_k_f.transpose(-2, -1))  # shape: (B, H, M, N)

    # Compute sliding window (sparse) attention scores.
    a_sm = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # shape: (B, H, M, N)
    a_sm_masked = a_sm.masked_fill(~mask_window.bool(), mask_value)
    a_sm_max = a_sm_masked.amax(dim=-1, keepdim=True)
    a_sm_exp = torch.exp(a_sm - a_sm_max)

    # Compute scores exactly as in the old implementation.
    # Sum over the query dimension (dim=-2) with keepdim and broadcast by mask_linear.

    #w_mask = mask_window * torch.ones((M, N), device=q.device).tril(M-N-window_size)[None, None, ...] #this should be like no sliding window.
    #w_mask instead of mask_window
    scores = torch.sum(torch.abs(a_ln - a_sm_exp) * mask_window, dim=-2, keepdim=True)
    local_scores = scores[:,:,0,-window_size:] #Return the last window-size many scores for generation.
    scores = scores * mask_linear
    #scores = scores / (torch.sum(mask_window, dim=-2, keepdim=True)+eps) #NOTE: GET THE AVERAGE SCORE, not the sum of all of them!
    # Now scores has shape (B, H, 1, N) broadcasted to (B, H, M, N).

    topk = min(global_cache_size, k.size(-2))  # k.size(-2) == N
    _, topk_indices = torch.topk(scores, k=topk, dim=-1)
    # Create a sparse mask of the same shape as scores.
    mask_sparse = torch.zeros_like(scores, dtype=torch.int, device=q.device)
    mask_sparse.scatter_(-1, topk_indices, 1)

    mask_sparse = mask_sparse * mask_linear  # elementwise multiplication

    # Update masks as in the original.
    mask_window = mask_window + mask_sparse
    mask_linear = mask_linear - mask_sparse

    # ---------------------------
    # Compute final attention components using updated masks.
    # ---------------------------
    # 1. Sliding window (softmax) attention
    a_sm = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
    a_sm_max = a_sm.amax(dim=-1, keepdim=True)
    a_sm = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # 2. Linear (low-rank) attention
    a_ln = torch.matmul(f_q_f, f_k_f.transpose(-2, -1))
    a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0)
    sum_ln = a_ln.sum(dim=-1, keepdim=True)

    # 3. Combine to compute attention weights.
    a = ((a_sm + a_ln) / (sum_sm + sum_ln + eps)).to(q.dtype)

    # Compute output.
    y = torch.matmul(a_sm + a_ln, v.float())
    if kv_state is not None:
        y += linear_factor * torch.matmul(f_q_f, kv_state.float())
        sum_ln = sum_ln + linear_factor * torch.matmul(f_q_f, k_state.float().unsqueeze(-1))
    y = (y / (sum_sm + sum_ln + eps)).to(q.dtype)

    if return_global_cache:
        if N > window_size:
            B,H,_,d = k_f.size()
            device = k_f.device
            B_idx = torch.arange(B,device=device)[:,None,None] # shape [B,1]
            H_idx = torch.arange(H,device=device)[None,:,None] # shape [1,H]

            global_scores, top_indices = torch.topk(scores[:,:,-1], k=topk, dim=-1)
            global_cache = (local_scores, global_scores, k_f[B_idx,H_idx,top_indices], v[B_idx,H_idx,top_indices])
        else:
            global_cache = (local_scores, None, None, None)
        return y, a, global_cache
    
    return y, a


# ---------------------
# Attention layer class
# ---------------------
class LolcatsSparsePrefillSlidingWindowAttention(LolcatsLinearAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """
    def __init__(self, 
                 window_size: int = 64, 
                 decode_window_size: int = None,
                 affine_attention_factors: bool = False,
                 init_window_factor: float = 0,
                 train_window_factor: bool = True,
                 state_grad_enabled: bool = False,
                 global_cache_size: int = 16,
                 **kwargs):
        self.window_size = window_size
        self.global_cache_size = global_cache_size
        self.decode_window_size = (
            decode_window_size if decode_window_size is not None else window_size
        )
        self.window_kwargs = {'dimension': 2, 'size': window_size, 'step': 1}
        super().__init__(**kwargs)
        self.attention_type = kwargs['attention_type']  #  'hedgehog_llama_window_sw'
        # Determine how we compute attentions
        self.quadratic_attention = hybrid_attention_quadratic
        self.attention_type = kwargs['attention_type']  # 'hedgehog_long_llama_window_sw'
        # Learnable factor for combining attentions
        self.affine_attention_factors = affine_attention_factors
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        if train_window_factor:
            self.window_factors = nn.Parameter(
                init_window_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype))
        else:
            self.register_buffer(
                "window_factors", init_window_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype)
            )
        # Whether we use original flash attention 2 inference (use during attention transfer)
        self.base_inference = False
        self.state_grad_enabled = state_grad_enabled
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with the option to compute attention weights multiple ways
        if self.train_attention is True
        -> Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        q, k, v, kv_seq_len = self.process_qkv(hidden_states, attention_mask, 
                                               position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)  # Have to do after repeat for grouped-query attn if we use same fmap

        if self.train_attention:
            # 1. Compute "ground-truth" attention output and weights
            with torch.no_grad():
                _y_true, a_true = softmax_attention(q, k, v)[:2]
                y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)

            # 2. Compute "predicted" attention outputs
            # compute attn weights under sliding window
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1
            y_pred, a_pred = self.quadratic_attention(q, k, f_q, f_k, v,
                                                      window_factors, linear_factors,
                                                      window_size=self.window_size,
                                                      global_cache_size=self.global_cache_size)
            attn_weights = ((a_pred, a_true), (y_pred, _y_true))
        else:
            attn_weights = None
            # attention_mask = None  # For now this is always True
            if past_key_value is None:  # Regular training
                #print('Training')
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                y_true, a_pred = self.quadratic_attention(q, k, f_q, f_k, v,
                                                          window_factors, linear_factors,
                                                          window_size=self.window_size,
                                                      global_cache_size=self.global_cache_size)
                attn_weights = a_pred
            else:
                past_key_value.window_size = self.decode_window_size
                past_key_value.global_cache_size = self.global_cache_size

                if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:  # Generating
                    #print('Generating', q.size(), f_q.size(), k.size(), f_k.size())
                    assert use_cache is True
                    _kv = past_key_value.update_for_decoding(k, v, self.layer_idx,
                                                             self.feature_map_k,
                                                             dtype=q.dtype)
                    local_k_cache, local_v_cache, global_k_cache, global_v_cache, f_kv_state, f_k_state = _kv

                    # Sliding window + linear attention decode
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = 1 - window_factors if self.affine_attention_factors else 1

                    # Softmax attention terms
                    if global_k_cache is not None:
                        k_cache = torch.cat([local_k_cache, global_k_cache], dim=-2)
                        v_cache = torch.cat([local_v_cache, global_v_cache], dim=-2)
                    else:
                        k_cache = local_k_cache
                        v_cache = local_v_cache

                    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
                    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                    a_sm   = window_factors * torch.exp(a_sm - a_sm_max)
                    sum_sm = a_sm.sum(dim=-1, keepdim=True)

                    #calculate errors for linear attention in sliding window 
                    f_k_sw = self.feature_map_k(k_cache[:,:,:self.window_size])
                    window_ln = torch.einsum('bhmd,bhnd->bhn', f_q.float(), f_k_sw.float())
                    local_errors = torch.abs(a_sm[:,:,0,:self.window_size]-window_ln)

                    past_key_value.update_scores(local_errors, self.layer_idx)
                    

                    # Combine with linear attention terms
                    y_true = (torch.einsum('bhmn,bhnd->bhmd', a_sm, v_cache.float())
                              + linear_factors * torch.einsum('bhlf,bhfd->bhld', f_q.float(), f_kv_state.float()))
                    sum_ln = linear_factors * torch.einsum(
                        'bhlf,bhnf->bhl', f_q.float(), f_k_state.float())[..., None]
                    y_true = (y_true / (sum_sm + sum_ln)).to(q.dtype) 

                else:  # Stateful training
                    #print('Stateful Training', q.size(), f_q.size(), k.size(), f_k.size())
                    try:
                        kv_state = past_key_value.kv_states[self.layer_idx]
                        k_state  = past_key_value.k_states[self.layer_idx]
                    except IndexError:
                        kv_state, k_state = None, None
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                    y_true, _, global_cache = self.quadratic_attention(q, k, f_q, f_k, v,
                                                         window_factors, linear_factors,
                                                         window_size=self.window_size,
                                                         kv_state=kv_state,
                                                         k_state=k_state,
                                                         global_cache_size=self.global_cache_size,
                                                         return_global_cache=True)

                    # Save and update KV cache and states
                    # past_key_value.update(k, v.detach(), self.layer_idx,
                    #                       fmap_key_states=f_k.detach(),
                    #                       accumulate_in_fp32=True)
                    past_key_value.update(k, v, self.layer_idx,
                                          fmap_key_states=f_k,
                                          accumulate_in_fp32=True,
                                          global_cache=global_cache)
                    
            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)

        return y_true, attn_weights, past_key_value


class LinearAttentionSparseSlidingWindowCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self, window_size: int = 64, global_cache_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states:  List[torch.Tensor] = []

        # Account for sliding windows
        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.window_size = window_size

        # Sparse Cache
        self.local_scores: List[torch.Tensor] = []
        self.global_scores: List[torch.Tensor] = []
        self.global_k_cache: List[torch.Tensor] = []
        self.global_v_cache: List[torch.Tensor] = []
        self.global_cache_size = global_cache_size

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: Optional[int] = None, cache_kwargs: Optional[any] = None,
               accumulate_in_fp32: bool = False, 
               fmap_key_states: torch.Tensor = None,  # should not be None
               grad_enabled: bool = False,
               global_cache: torch.Tensor = None,
               **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV, K states; and KV cache during training
        - For decoding, use `self.decode_kv_states` to keep track of KV states 
          up to sliding window terms
        - For (chunked) training, use `self.kv_states` to keep track of KV states
          up to end of sequence
        - Likewise for `self.decode_k_states` and `self.k_states`
        """
        with torch.set_grad_enabled(grad_enabled):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            dtype = key_states.dtype
            if accumulate_in_fp32:
                # key_states = key_states.float()
                fmap_key_states = fmap_key_states.float()
                value_states = value_states.float()

            # Decoding KV state (KV terms up to last window_size)
            decode_kv_state = torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states[:, :, :-self.window_size], value_states[:, :, :-self.window_size]
            )

            # KV state
            kv_state = decode_kv_state + torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states[:, :, -self.window_size:], value_states[:, :, -self.window_size:]
            )
            # shape is b, h, 1, f; note the 1
            decode_k_state = fmap_key_states[:, :, :-self.window_size].sum(dim=-2, keepdim=True)
            k_state = (decode_k_state + fmap_key_states[:, :, -self.window_size:].sum(dim=-2, keepdim=True))

            local_scores, global_scores, global_k_cache, global_v_cache = global_cache

            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))

                self.decode_kv_states.append(decode_kv_state.to(dtype))
                self.decode_k_states.append(decode_k_state.to(dtype))

                self.k_cache.append(key_states[:, :, -self.window_size:, :].to(dtype))
                self.v_cache.append(value_states[:, :, -self.window_size:, :].to(dtype))
                # self._seen_tokens_by_layer[layer_idx].append(key_states.shape[-2])

                self.local_scores.append(local_scores.to(dtype))

                if key_states.size(2) > self.window_size: #Otherwise, global_scores, etc. are None.
                    self.global_scores.append(global_scores.to(dtype))
                    self.global_k_cache.append(global_k_cache.to(dtype))
                    self.global_v_cache.append(global_v_cache.to(dtype))

            else:
                print('[OOPS!] Its updating recurrently!, havent touched this code! ')
                
                # Update kv and k states recurrently
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(dtype)
                k_state  = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(dtype)
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx]  = k_state

                decode_kv_state = (self.decode_kv_states[layer_idx].to(kv_state.dtype) 
                                   + decode_kv_state).to(dtype)
                decode_k_state  = (self.decode_k_states[layer_idx].to(kv_state.dtype) 
                                   + decode_k_state).to(dtype)
                self.decode_kv_states[layer_idx] = decode_kv_state
                self.decode_k_states[layer_idx]  = decode_k_state

                self.k_cache[layer_idx] = key_states[:, :, -self.window_size:, :]
                self.v_cache[layer_idx] = value_states[:, :, -self.window_size:, :]

            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def update_for_decoding(self, keys: torch.Tensor, values: torch.Tensor, 
                            layer_idx: int, feature_map_k: Callable, dtype: torch.dtype):
        """
        Update the decoding KV and K states, and KV cache, during decodeing
        """
        with torch.no_grad():
            k_cache = self.k_cache[layer_idx]
            v_cache = self.v_cache[layer_idx]

            #If local cache has space, add it here!
            if k_cache.shape[-2] < self.window_size:  # build window-size cache
                self.k_cache[layer_idx] = torch.cat([k_cache, keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache, values], dim=-2)

                B,H,S = self.local_scores[layer_idx].size()
                empty_score = torch.zeros((B,H,1), device=self.local_scores[layer_idx].device, dtype=dtype)
                self.local_scores[layer_idx] = torch.cat([self.local_scores[layer_idx], empty_score], dim=-1)

            #
            elif len(self.global_k_cache) <= layer_idx:
                self.global_k_cache.append(k_cache[:, :, :1, :])
                self.global_v_cache.append(v_cache[:, :, :1, :])

                self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)

                self.global_scores.append(self.local_scores[layer_idx][:,:,:1])
                self.local_scores[layer_idx] = torch.roll(self.local_scores[layer_idx], -1, dims=-1)
                self.local_scores[layer_idx][:, :, -1] = 0

            elif self.global_k_cache[layer_idx].shape[-2] < self.global_cache_size: # build  global_cache_size cache
                #Grab last kv from sliding window, add it to global_sparse_cache
                self.global_k_cache[layer_idx] = torch.cat([self.global_k_cache[layer_idx], k_cache[:, :, :1, :]], dim=-2)
                self.global_v_cache[layer_idx] = torch.cat([self.global_v_cache[layer_idx], v_cache[:, :, :1, :]], dim=-2)
                self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)

                self.global_scores[layer_idx] = torch.cat([self.global_scores[layer_idx], self.local_scores[layer_idx][:,:,:1]], dim=-1)
                self.local_scores[layer_idx] = torch.roll(self.local_scores[layer_idx], -1, dims=-1)
                self.local_scores[layer_idx][:, :, -1] = 0

            else:
                # MZ 6/3: handle short inputs; zero-out padding when initial k.shape[2] < self.window_size
                # if k_cache[:, :, :1, :].sum() == 0:   # heuristic for zeroing out padding in cache
                #     f_k_state = torch.zeros(k_cache[:, :, :1, :].shape, dtype=dtype, device=k_cache.device)
                # else:
                #     f_k_state = feature_map_k(k_cache[:, :, :1, :])
                # -> MZ (later): above only relevant if we zero-pad in our hybrid attention computation
                
                """ Update the leaderboard with the new high scoring KV pairs """
                B, H, G = self.global_scores[layer_idx].size()
                device = self.global_scores[layer_idx].device
                B_idx = torch.arange(B,device=device).unsqueeze(-1) # shape [B,1]
                H_idx = torch.arange(H,device=device).unsqueeze(0) # shape [1,H]

                min_scores, min_indices = torch.min(self.global_scores[layer_idx], dim=-1) #shape [B,H]
                mask = (self.local_scores[layer_idx][:,:,0] < min_scores) #shape (B,H)

                min_global_score = self.global_scores[layer_idx][B_idx, H_idx, min_indices] # shape [B,H]
                high_scores = torch.where(mask, min_global_score, self.local_scores[layer_idx][:,:,0]) # shape [B,H]

                #self.global_scores[layer_idx].scatter(dim=2, index=min_indices,src=high_scores)
                self.global_scores[layer_idx][B_idx, H_idx, min_indices] = high_scores
            

                """ Replace the high scoring KV pairs in the global cache """           
                #min_indices = min_indices.unsqueeze(-1).expand(-1,-1,-1,self.global_k_cache[layer_idx].size(-1))
                min_global_keys = self.global_k_cache[layer_idx][B_idx, H_idx, min_indices]
                min_global_values = self.global_v_cache[layer_idx][B_idx, H_idx, min_indices]

                
                    
                mask=mask[...,None]

                easy_keys   = torch.where(mask, k_cache[:,:,0,:], min_global_keys)
                easy_values = torch.where(mask, v_cache[:,:,0,:], min_global_values)
                difficult_keys   = torch.where(mask, min_global_keys, k_cache[:,:,0,:])
                difficult_values = torch.where(mask, min_global_values, v_cache[:,:,0,:])
                
                self.global_k_cache[layer_idx][B_idx, H_idx, min_indices] = difficult_keys
                self.global_v_cache[layer_idx][B_idx, H_idx, min_indices] = difficult_values
                
                """ Place easy KV pairs in the hidden state """           
                k_state = feature_map_k(easy_keys.unsqueeze(-2).to(dtype))
                v_state = easy_values.unsqueeze(-2).to(dtype)
                kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(dtype) # b, h, f, d
                self.decode_kv_states[layer_idx] += kv_state
                self.decode_k_states[layer_idx] += k_state
                
                """ Shift Sliding Window """  
                self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)
                self.local_scores[layer_idx] = torch.roll(self.local_scores[layer_idx], -1, dims=-1)
                self.local_scores[layer_idx][:, :, -1] = 0
            
            if layer_idx == 0:
                self._seen_tokens += keys.shape[-2]
            self._seen_tokens_by_layer[layer_idx] += keys.shape[-2]
            if len(self.global_k_cache) <= layer_idx:
                return (self.k_cache[layer_idx], self.v_cache[layer_idx], 
                        None, None,
                        self.decode_kv_states[layer_idx], self.decode_k_states[layer_idx])
            else:
                return (self.k_cache[layer_idx], self.v_cache[layer_idx], 
                        self.global_k_cache[layer_idx], self.global_v_cache[layer_idx],
                        self.decode_kv_states[layer_idx], self.decode_k_states[layer_idx])


    
    def update_scores(self, scores, layer_idx):
        #self.local_scores[layer_idx] is shape (B,H,len(k_cache))
        self.local_scores[layer_idx][:,:,:scores.size(2)] += scores



"""
NOTEPAD FOR ADDING SPARSE CACHE

6. update
- need to replace with finite memory attention 
"""
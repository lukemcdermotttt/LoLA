#!pip install flash-attn --no-build-isolation
#from flash_attn import flash_attn_func
import torch
#import time
from einops import rearrange 

B,N,H,D = 2,4096,4,96
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
#torch.manual_seed(42)
window_size=32
global_cache_size=128

q = torch.randn((B,N,H,D),device=device,dtype=dtype)
k = torch.randn((B,N,H,D),device=device,dtype=dtype)
v = torch.randn((B,N,H,D),device=device,dtype=dtype)
f_q = torch.randn((B,N,H,2*D),device=device,dtype=dtype)
f_k = torch.randn((B,N,H,2*D),device=device,dtype=dtype)
scale = q.size(-1)**(.5)
chunk_size = window_size

q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size)
k = rearrange(k, 'b (n c) h d -> b h n c d', c=chunk_size)
v = rearrange(v, 'b (n c) h d -> b h n c d', c=chunk_size)
f_q = rearrange(f_q, 'b (n c) h d -> b h n c d', c=chunk_size)
f_k = rearrange(f_k, 'b (n c) h d -> b h n c d', c=chunk_size)

_ , _ , n, c, _ = q.size()

#Step 1 Diagonal Chunks
qk = torch.tril(q @ k.transpose(-1,-2)) / scale #shape (b h n c_q c_k), but its lower triangle
safe_max = torch.amax(qk, dim=-1).unsqueeze(-1) / 2 #shape (b h n c_q 1), this is a relatively large constant that we can divide other stuff with e^safe_max
exp_qk = torch.exp(qk-safe_max)
phi_qk = torch.tril(f_q @ f_k.transpose(-1,-2)) / torch.exp(safe_max) #shape (b h n c_q c_k), also lower triangle.

#Outputs of this Step, everything above can be overwritten
scores = torch.sum((exp_qk-phi_qk)**2, dim=-2) #shape (b h n c_k)
y = exp_qk @ v #shape (b h n c_q d)
denom = torch.sum(exp_qk, dim=-1) #shape (b h n c_q)

#Step 2 Off-Diagonal Chunks
qk = torch.cat([ torch.zeros_like(q[:,:,:1]), q[:,:,1:] ],dim=2) @ k.transpose(-1,-2) / scale #Roll q & matmul keys. Dense Matrix, no tril or triu yet.
phi_qk = torch.cat([ torch.zeros_like(f_q[:,:,:1]), f_q[:,:,1:] ],dim=2) @ f_k.transpose(-1,-2)

rolled_safe_max = torch.roll(safe_max, 1, 2)
exp_qk = torch.exp(qk-rolled_safe_max)
exp_qk[:,:,:1] *= 0
phi_qk /= rolled_safe_max

#Outputs of this Step, everything above can be overwritten
scores += torch.sum(torch.triu(exp_qk-phi_qk)**2, dim=-2)
y += torch.roll(exp_qk @ v, -1, 2) #shape (b h n c_q d)
denom += torch.roll(torch.sum(exp_qk, dim=-1),-1,2) #shape (b h n c_q)

# --- Step 3: rolling FIFO cache + sparse global attention ---

# flatten k, f_k, v for easy B,H,seq indexing
k_flat   = rearrange(k,   'b h n c d -> b h (n c) d')
f_k_flat = rearrange(f_k, 'b h n c d -> b h (n c) d')
v_flat   = rearrange(v,   'b h n c d -> b h (n c) d')

lambda_chunks = global_cache_size // chunk_size
fill_end      = 2 + lambda_chunks

# initialize (will get overwritten on first iteration)
cache_positions = torch.arange(global_cache_size, device=device)

for q_chunk_idx in range(2, n):
    # pick out this chunk’s q, f_q, and its safe‐max
    q_chunk       = q[:,   :, q_chunk_idx]        # (B, H, c, d)
    f_q_chunk     = f_q[:, :, q_chunk_idx]        # (B, H, c, 2*d)
    safe_max_chunk = safe_max[:, :, q_chunk_idx]  # (B, H, c, 1)

    # fill phase: grow cache until it reaches global_cache_size
    if q_chunk_idx <= fill_end:
        cur_size = min((q_chunk_idx - 1) * chunk_size, global_cache_size)
        cache_positions = torch.arange(cur_size, device=device)
    else:
        # rolling FIFO: drop oldest chunk_size, append the latest chunk
        start      = (q_chunk_idx - 2) * chunk_size
        new_pos    = torch.arange(start, start + chunk_size, device=device)
        cache_positions = torch.cat([cache_positions, new_pos], dim=0)[chunk_size:]

    # gather the cached keys & values
    k_cache   = k_flat[:,   :, cache_positions]  # (B, H, cache_size, d)
    f_k_cache = f_k_flat[:, :, cache_positions]  # (B, H, cache_size, 2*d)
    v_cache   = v_flat[:,   :, cache_positions]  # (B, H, cache_size, d)

    # full‐attention vs linear‐attention terms
    qk_global      = (q_chunk @ k_cache.transpose(-1, -2)) / scale
    phi_global     = (f_q_chunk @ f_k_cache.transpose(-1, -2)) / torch.exp(safe_max_chunk)
    exp_qk_global  = torch.exp(qk_global - safe_max_chunk)

    # accumulate into y and denom at this chunk index
    y[:, :, q_chunk_idx:q_chunk_idx+1]   += (exp_qk_global @ v_cache).unsqueeze(2)
    denom[:, :, q_chunk_idx:q_chunk_idx+1] += torch.sum(exp_qk_global, dim=-1).unsqueeze(2)

output = y / denom.unsqueeze(-1)






"""
Outline:

SWA Attention + Key-Scores. For fast training, the true sliding window size is in between [eta, 2*eta).
1. Diagonal Chunks
    - Compute Attn & Lin. Attn for scores w/ Masking off upper triangle
    - Use safe softmax, but use the max/2. store the max somewhere so we can normalize everything. Shape (b,h,n,c)
    - Divide phi(q)'s by max/2, so that all exp(qk) and phi(q)phi(k) have the same division.
    - Mult exp(qk) by values and sum over values to store in y.
    - Store scores in a running sum of shape (b,h,n,c) representing key indices
    - Do not keep lin. attn outputs.
2. Off-diagonal Chunk 
    - Compute Attn & Lin. Attn for scores. Only use masking for scores by masking off lower triangle
    - Use the safe softmax max from the diagonal (max/2)
    - Mult exp(qk) by values, add to running sum for y.
    - Do not keep lin. attn outputs here.
3. Accumulate & Finalzie SWA Outputs and Scores
    - y is shape (b,h,n,c,d), scores is shape (b,h,n,c)
4. Chunk Recurrent Calculation for Scores
    - The top left (2 + lambda/chunk_size) chunk grid should be pure attention.
    - For the (3 + lambda/chunk_size)th query chunk:
        - create a "cache" that contains first (1 + lambda/chunk_size) chunks of keys
            - lambda/chunk_size is "# chunks in past, real cache", +1 is "# chunks with potential new keys"
        - compute exp(qk) on the cache using safe exp with m/2
        - calculate the top lambda scores from the first lambda+chunk_size indices (aka 1+lambda/chunk many chunks of keys)
        - mask off exp(qk) for scores that arent included.
        - update cache with new indices
    - iterate next query chunk, same thing except the cache is already created ...
"""












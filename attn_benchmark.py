#!pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func
import torch
import time
from einops import rearrange 
B,N,H,D = 32,4096,32,64
num_iters = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
torch.manual_seed(42)

q = torch.randn((B,N,H,D),device=device,dtype=dtype)
k = torch.randn((B,N,H,D),device=device,dtype=dtype)
v = torch.randn((B,N,H,D),device=device,dtype=dtype)
print(q.size())

for _ in range(10):
    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True,window_size=(64,64))
start_time = time.time()
for _ in range(num_iters):
    out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True,window_size=(64,64))
end_time = time.time()

print('Flash Attention, Total Time: ', end_time - start_time)
print()
chunk_size=64
q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size)
k = rearrange(k, 'b (n c) h d -> b h n c d', c=chunk_size)
v = rearrange(v, 'b (n c) h d -> b h n c d', c=chunk_size)
print(q.size())

for _ in range(10):
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    intra = ((
        q @ k.transpose(-1, -2)).masked_fill_(
        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
        0
    )) @ v
    o = inter + intra
torch.cuda.synchronize()
start_time = time.time()
for _ in range(num_iters):
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    intra = ((
        q @ k.transpose(-1, -2)).masked_fill_(
        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
        0
    )) @ v
    o = inter + intra
torch.cuda.synchronize()
end_time = time.time()

print('Linear Attention, Total Time: ', end_time - start_time)
print(o.size())
# lola_fast.py  –  end‑to‑end LoLA sparse attention (window + residual‑topk + linear)
# -----------------------------------------------------------------------------------
#   pip install flash-attn --no-build-isolation
#   pip install triton==2.1.0  (or latest released version)

import torch, triton, triton.language as tl, time
from einops import rearrange
from flash_attn import flash_attn_func

# --- remove proj_topk_kernel and replace lola_sparse with: ------------------
def lola_sparse(q, k, v, fq, fk, *, chunk=64, cache=128):
    B,L,H,D = q.shape
    C,G     = chunk, cache
    N       = L // C
    scale   = D**-0.5
    dev     = q.device

    q_c  = rearrange(q,  'b (n c) h d -> b h n c d', c=C)
    k_c  = rearrange(k,  'b (n c) h d -> b h n c d', c=C)
    v_c  = rearrange(v,  'b (n c) h d -> b h n c d', c=C)
    fq_c = rearrange(fq, 'b (n c) h f -> b h n c f', c=C)
    fk_c = rearrange(fk, 'b (n c) h f -> b h n c f', c=C)

    k_flat  = rearrange(k_c,  'b h n c d -> b h (n c) d')
    v_flat  = rearrange(v_c,  'b h n c d -> b h (n c) d')
    fk_flat = rearrange(fk_c, 'b h n c f -> b h (n c) f')

    K_cache  = torch.zeros((B,H,G,D),   device=dev, dtype=q.dtype)
    V_cache  = torch.zeros_like(K_cache)
    FK_cache = torch.zeros((B,H,G,2*D), device=dev, dtype=q.dtype)

    S_h = torch.zeros((B,H,2*D,D), device=dev, dtype=torch.float32)
    S_s = torch.zeros((B,H,2*D),   device=dev, dtype=torch.float32)

    out   = torch.empty_like(q)
    denom = torch.empty((B,L,H), device=dev, dtype=torch.float32)

    for n in range(N):
        qn, kn, vn = q_c[:,:,n], k_c[:,:,n], v_c[:,:,n]
        fqn, fkn   = fq_c[:,:,n], fk_c[:,:,n]
        km1 = k_c[:,:,n-1] if n else torch.zeros_like(kn)
        vm1 = v_c[:,:,n-1] if n else torch.zeros_like(vn)

        K_block = torch.cat([km1, kn, K_cache], dim=2)
        V_block = torch.cat([vm1, vn, V_cache], dim=2)

        yw = flash_attn_func(
            rearrange(qn,'b h c d->(b h) c 1 d'),
            rearrange(K_block,'b h s d->(b h) s 1 d'),
            rearrange(V_block,'b h s d->(b h) s 1 d'),
            causal=False
        )
        yw = rearrange(yw,'(b h) c 1 d->b h c d', b=B, h=H)
        qk = torch.einsum('b h c d, b h s d->b h c s', qn, K_block)*scale
        den_w = torch.exp(qk - qk.amax(-1,keepdim=True)).sum(-1)

        y_lin  = torch.einsum('b h c f, b h f d->b h c d', fqn.float(), S_h)
        den_lin= torch.einsum('b h c f, b h f  -> b h c',  fqn.float(), S_s)

        resid = (yw - y_lin).mean(2)                        # [B,H,D]
        seen  = (n+1)*C
        score = torch.einsum('b h d, b h l d->b h l', resid.float(),
                             v_flat[:,:,:seen].float()).abs()  # [B,H,seen]
        topk  = score.topk(min(G, seen), dim=-1)

        idx   = topk.indices                                 # [B,H,G']
        bidx  = torch.arange(B,device=dev)[:,None,None]
        hidx  = torch.arange(H,device=dev)[None,:,None]

        K_cache = k_flat[bidx,hidx,idx].to(q.dtype)
        V_cache = v_flat[bidx,hidx,idx].to(q.dtype)
        FK_cache= fk_flat[bidx,hidx,idx].to(q.dtype)

        S_h = torch.einsum('b h g f, b h g d->b h f d', FK_cache.float(),
                                                          V_cache.float())
        S_s = FK_cache.float().sum(2)

        s0,s1 = n*C,(n+1)*C
        out[:,s0:s1]   = rearrange(yw+y_lin,'b h c d->b c h d')
        denom[:,s0:s1] = (den_w+den_lin).permute(0,2,1)

    return out / denom.unsqueeze(-1)


# -----------------------------------------------------------------------------------
def benchmark():
    B,L,H,D = 32,4096,32,64
    C,G     = 64,128
    dev     = 'cuda'
    dtype   = torch.bfloat16
    iters   = 50

    torch.manual_seed(0)
    q  = torch.randn((B,L,H,D),    device=dev, dtype=dtype)
    k  = torch.randn_like(q)
    v  = torch.randn_like(q)
    fq = torch.randn((B,L,H,2*D),  device=dev, dtype=dtype)
    fk = torch.randn_like(fq)

    # warm‑up
    for _ in range(3):
        _ = lola_sparse(q,k,v,fq,fk,chunk=C,cache=G)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = lola_sparse(q,k,v,fq,fk,chunk=C,cache=G)
    torch.cuda.synchronize()
    print(f"LoLA‑Fast  {iters} iters: {time.time()-t0:.3f}s  "
          f"({(time.time()-t0)/iters*1000:.2f} ms/iter)")
    print("out:", _.shape)

if __name__ == "__main__":
    benchmark()

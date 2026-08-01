"""Op-by-op: is each ttnn op in the MSA path invariant to chunking the depth axis?

Result: layer_norm, linear, the token-axis matmul, and a concat-built tensor (data AND memory_config)
are all BIT-EXACT under depth chunking, at the shapes tried here.

Read that carefully -- it is a narrower statement than it looks, and over-reading it cost several
passes. Component-level bit-exactness does NOT compose into fold-level bit-exactness: every op here
passed while the full fold still differed, because the difference lives in shapes these isolated
cases did not reproduce. Probe shapes must match production shapes, non-alignment included.

The hypothesis this file was written to test -- that a ttnn matmul's per-core K-blocking is re-planned
when the shape changes, so PWA's token-axis reduction gets re-blocked -- is REFUTED by check 3: the
matmul is bit-exact depth-chunked with the auto-selected config. The pinned-program_config path is
kept because it is the tool if that question ever comes back, not because it was needed.

Run on the Galaxy with one chip free:
    TT_VISIBLE_DEVICES=<n> python3 -u scripts/abag_xm/probe_chunk_bitexactness.py
"""
import os
import torch
import ttnn

from tt_bio.tenstorrent import get_device, MSA_CHUNK_SIZE, CORE_GRID_MAIN

D = int(os.environ.get("PROBE_DEPTH", 8998))   # MSA depth (9lwc's real depth)
S = int(os.environ.get("PROBE_TOKENS", 288))   # padded token count
C = 128                                        # c_m
HD = 8                                         # PWA head_dim
CHUNK = MSA_CHUNK_SIZE


def up(t, dev):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def report(name, whole, chunked):
    """whole/chunked are torch tensors already pulled back to host."""
    same = torch.equal(whole, chunked)
    if same:
        print(f"  {name:38s} BIT-EXACT")
        return True
    d = (whole.float() - chunked.float()).abs()
    nz = (d > 0).float().mean().item()
    print(f"  {name:38s} DIFFERS  maxabs {d.max():.3e}  frac_diff {nz:.4f}")
    return False


def main():
    dev = get_device()
    torch.manual_seed(0)
    print(f"shapes: m=({D},{S},{C})  w=(1,{S},{S})  chunk={CHUNK}  head_dim={HD}")

    m_t = torch.randn(D, S, C, dtype=torch.float32)
    w_t = torch.randn(1, S, S, dtype=torch.float32)
    ln_w = torch.randn(C, dtype=torch.float32)
    ln_b = torch.randn(C, dtype=torch.float32)
    proj = torch.randn(C, HD, dtype=torch.float32)

    m = up(m_t, dev)
    w = up(w_t, dev)
    lnw, lnb = up(ln_w, dev), up(ln_b, dev)
    pw = up(proj, dev)

    def slices(t):
        return [t[s:min(s + CHUNK, D)] for s in range(0, D, CHUNK)]

    # 1. layer_norm -- normalises over C, so depth extent should be irrelevant
    whole = ttnn.to_torch(ttnn.layer_norm(m, weight=lnw, bias=lnb, epsilon=1e-5))
    parts = [ttnn.layer_norm(c, weight=lnw, bias=lnb, epsilon=1e-5) for c in slices(m)]
    report("layer_norm(depth-chunked)", whole, ttnn.to_torch(ttnn.concat(parts, dim=0)))

    # 2. linear (the c=128 -> head_dim projection) -- reduces over C, not depth
    whole = ttnn.to_torch(ttnn.linear(m, pw, core_grid=CORE_GRID_MAIN))
    parts = [ttnn.linear(c, pw, core_grid=CORE_GRID_MAIN) for c in slices(m)]
    report("linear(depth-chunked)", whole, ttnn.to_torch(ttnn.concat(parts, dim=0)))

    # 3. THE SUSPECT: matmul contracting the TOKEN axis, batched over depth.
    #    v is (D, HD, S); w is (1, S, S); transpose_b contracts S.
    v_t = torch.randn(D, HD, S, dtype=torch.float32)
    v = up(v_t, dev)
    whole = ttnn.to_torch(ttnn.matmul(v, w, transpose_b=True))
    parts = [ttnn.matmul(c, w, transpose_b=True) for c in slices(v)]
    matmul_exact = report("matmul(depth-chunked, auto config)", whole,
                          ttnn.to_torch(ttnn.concat(parts, dim=0)))

    # 4. Same matmul with the K-blocking PINNED, so the token-axis decomposition cannot be
    #    re-planned when the batch extent changes. in0_block_w is the K-block width, in tiles.
    if not matmul_exact:
        gx, gy = CORE_GRID_MAIN.x, CORE_GRID_MAIN.y
        for in0_block_w in (1, S // 32):
            try:
                cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=in0_block_w,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=1,
                    out_block_w=max(1, S // 32),
                    per_core_M=1,
                    per_core_N=max(1, S // 32),
                    fuse_batch=False,
                    transpose_mcast=False,
                    fused_activation=None,
                )
                whole = ttnn.to_torch(ttnn.matmul(v, w, transpose_b=True, program_config=cfg))
                parts = [ttnn.matmul(c, w, transpose_b=True, program_config=cfg)
                         for c in slices(v)]
                report(f"matmul(pinned in0_block_w={in0_block_w})", whole,
                       ttnn.to_torch(ttnn.concat(parts, dim=0)))
            except Exception as e:                      # noqa: BLE001 -- probe, report and move on
                print(f"  matmul(pinned in0_block_w={in0_block_w}) CONFIG REJECTED: "
                      f"{str(e).splitlines()[0][:110]}")

    probe_concat_layout()
    print("done")




def probe_concat_layout():
    """Does a concat-built tensor behave identically to a monolithic one with the SAME bytes?

    probe 1-3 showed the per-op arithmetic is bit-exact under depth chunking, which refutes the
    "matmul K-blocking is re-planned by shape" theory. The remaining structural difference in the
    real fold is that the chunked path hands DOWNSTREAM ops a ttnn.concat result instead of a
    single allocation. If the concat's data is bit-identical but a later op on it is not, the
    divergence is a memory-config/layout effect, not arithmetic.
    """
    dev = get_device()
    torch.manual_seed(0)
    print(f"\nconcat-layout probe: m=({D},{S},{C})")
    m_t = torch.randn(D, S, C, dtype=torch.float32)
    ln_w, ln_b = torch.randn(C), torch.randn(C)
    mono = up(m_t, dev)
    lnw, lnb = up(ln_w, dev), up(ln_b, dev)

    built = ttnn.concat([mono[s:min(s + CHUNK, D)] for s in range(0, D, CHUNK)], dim=0)
    report("concat-built tensor vs monolithic (data)",
           ttnn.to_torch(mono), ttnn.to_torch(built))
    print(f"    mono  memory_config={mono.memory_config()}")
    print(f"    built memory_config={built.memory_config()}")

    report("layer_norm on concat-built vs monolithic",
           ttnn.to_torch(ttnn.layer_norm(mono, weight=lnw, bias=lnb, epsilon=1e-5)),
           ttnn.to_torch(ttnn.layer_norm(built, weight=lnw, bias=lnb, epsilon=1e-5)))


if __name__ == "__main__":
    main()

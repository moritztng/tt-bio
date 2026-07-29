"""What ttnn actually does with the tile padding of a non-tile-aligned TILE_LAYOUT tensor.

p21 inferred, end-to-end, that `ttnn.full` leaves the padding holding stale DRAM and that a
downstream softmax reads it. This settles it at op level instead, and the answer is the
other way round:

  * `ttnn.full` / `ttnn.zeros` fill the WHOLE padded buffer, padding included.
  * `ttnn.from_torch` (host upload) writes 0 into the padding, not the fill value.
  * `ttnn.matmul` does not contract the padding of a non-aligned K.
  * `ttnn.softmax(dim=-1)` does not include the padding in its denominator.

Reading the padding needs a trick: `ttnn.to_torch`, `ttnn.untilize` and `ttnn.view` all unpad
(`view` rejects the volume change outright). Allocating an *uninitialized* `ttnn.empty` of the
padded shape over the just-freed buffer, after checking `buffer_address()` matches, reads
exactly the bytes the tensor under test left behind.

  TT_VISIBLE_DEVICES=0 PYTHONPATH=<tree> python3 <tree>/scripts/rfd3_port/probe_tile_padding_semantics.py
"""

import torch
import ttnn

DIRTY, FILL = 7777.0, -1e4


def pad(n):
    return (n + 31) // 32 * 32


def padded_shape(shape):
    return (shape[0], shape[1], pad(shape[2]), pad(shape[3]))


def probe_constructors(dev, shape, label):
    """Dirty the buffer, build `shape` three ways, read back what each left in the padding."""
    ps = padded_shape(shape)
    for how, make in (
        ("ttnn.full", lambda: ttnn.full(shape, FILL, dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=dev)),
        ("ttnn.zeros", lambda: ttnn.zeros(shape, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=dev)),
        ("upload", lambda: ttnn.from_torch(torch.full(shape, FILL, dtype=torch.bfloat16),
                                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=dev)),
    ):
        d = ttnn.full(ps, DIRTY, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ad = d.buffer_address()
        ttnn.deallocate(d)
        t = make()
        at = t.buffer_address()
        ttnn.deallocate(t)
        e = ttnn.empty(ps, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        ae = e.buffer_address()
        v = ttnn.to_torch(e).float()
        ttnn.deallocate(e)
        if not (ad == at == ae):
            print(f"{label:8s} {how:10s} address mismatch {ad:#x}/{at:#x}/{ae:#x} -- inconclusive")
            continue
        body = v[..., :shape[2], :shape[3]]
        cols = v[..., :shape[2], shape[3]:]
        rows = v[..., shape[2]:, :]

        def rng(x):
            return "n/a" if x.numel() == 0 else f"[{x.min().item():.1f},{x.max().item():.1f}]"

        ref = body.flatten()[0]
        clean = all((x.numel() == 0) or bool((x == ref).all()) for x in (cols, rows))
        print(f"{label:8s} {how:10s} body={rng(body)} pad-cols={rng(cols)} pad-rows={rng(rows)}"
              f"  -> padding {'matches body' if clean else 'DIFFERS from body'}")


def probe_matmul(dev, ckc):
    """ones @ ones over a non-aligned K: 65 if the padding is masked, 96 if it is contracted."""
    M, K, N = 32, 65, 32

    def mk(shape, how):
        if how == "full":
            return ttnn.full(shape, 1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev)
        return ttnn.from_torch(torch.ones(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)

    for ha in ("upload", "full"):
        for hb in ("upload", "full"):
            a, b = mk((1, 1, M, K), ha), mk((1, 1, K, N), hb)
            r = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc)).float()
            print(f"matmul A={ha:6s} B={hb:6s} -> {r.min().item():.1f} "
                  f"(correct 65, padding-contracted 96)")
            ttnn.deallocate(a)
            ttnn.deallocate(b)

    # the rfd3 atom-attention shape: attention [1,4,L,L] @ v [1,4,L,32], L = 2702 = 84*32 + 14
    L, H, HD = 2702, 4, 32
    a, b = mk((1, H, L, L), "full"), mk((1, H, L, HD), "full")
    r = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc)).float()
    print(f"matmul L=2702 full@full -> {r.min().item():.1f} "
          f"(correct 2702 up to bf16 rounding, padding-contracted 2720)")
    ttnn.deallocate(a)
    ttnn.deallocate(b)


def probe_softmax(dev):
    """All -1e4 logically, padding -1e4 vs 0. Identical => the padding is not in the sum."""
    L, H = 2702, 4
    shape = (1, H, L, L)
    f = ttnn.full(shape, FILL, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    sf = ttnn.to_torch(ttnn.softmax(f, dim=-1)).float()
    ttnn.deallocate(f)
    u = ttnn.from_torch(torch.full(shape, FILL, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    su = ttnn.to_torch(ttnn.softmax(u, dim=-1)).float()
    ttnn.deallocate(u)
    print(f"softmax(pad=-1e4) vs softmax(pad=0): maxabs {(sf - su).abs().max().item():.6e} "
          f"(0.0 => padding excluded from the denominator)")


def main():
    dev = ttnn.open_device(device_id=0)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           math_approx_mode=False, fp32_dest_acc_en=True,
                                           packer_l1_acc=True)
    try:
        probe_constructors(dev, (1, 1, 32, 65), "small")
        probe_constructors(dev, (1, 1, 256, 65), "zeroK")
        probe_constructors(dev, (1, 4, 2702, 2702), "mask")
        probe_matmul(dev, ckc)
        probe_softmax(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

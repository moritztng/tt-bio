#!/usr/bin/env python3
"""gap_dit_mask: the DiT's stacked SDPA mask build, shipped against forms that pad less.

`LocalTokenTransformer._dit_sdpa_masks` builds every block's mask per DiT call (2 calls/step) as one
[1, 288, n, n] stack: linear(z) -> permute -> multiply(sqrt d) -> add(additive mask) -> pad axis 3 -> pad axis 2,
then the DiT slices block i's 16 heads out. Each of those is a full pass over ~285 MB. Arms:
  shipped  as above
  pad2     the two stack pads as one ttnn.pad over both axes
  zpad     z padded with zeros to [1, n, n, 128] and the additive mask with -1e4 to [1, 1, n, n] first, so the
           stack comes out padded: pad rows/cols are 0 * sqrt(d) + -1e4 = -1e4, exactly what the pads write
  zpad_af  zpad with the sqrt(d) scale folded into the add as an input activation (one pass fewer)
Every arm also slices the 18 blocks, as the DiT does. Each arm is compared with shipped by torch.equal over the full
padded [1, 288, n, n] stack, and all against a float64 reference on the logical region.

    python scripts/rfd3_port/gap_dit_mask.py OUT.json [reps] [I]
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                      # noqa: E402

ensure_p300_mesh_descriptor()
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.tiles import pad_axis                                   # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_dit_mask.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
I = int(sys.argv[3]) if len(sys.argv) > 3 else 685
C, HB, NB, HD = 128, 16, 18, 48
H = HB * NB
NK = M.align_tile(I)


def timeit(fn, dev):
    o = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N):
        for x in o:
            ttnn.deallocate(x)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return o, statistics.median(ts), min(ts), max(ts)


def main():
    dev = get_device()
    torch.manual_seed(42)
    ckc = M._default_compute_kernel_config()
    lin = dict(ckc=ckc, dtype=ttnn.bfloat16, core_grid=M.CORE_GRID_MAIN)
    z_t = torch.randn(1, I, I, C).bfloat16()
    w_t = (torch.randn(C, H) / C ** 0.5).bfloat16()
    am_t = torch.where(torch.rand(1, 1, I, I) < 0.3, -1e4, 0.0).bfloat16()
    ref = (torch.einsum("bijc,ch->bhij", z_t.double(), w_t.double()) * HD ** 0.5 + am_t.double())
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z, w, am = up(z_t), up(w_t), up(am_t)

    def blocks(stack):
        out = [ttnn.slice(stack, [0, HB * i, 0, 0], [1, HB * (i + 1), NK, NK]) for i in range(NB)]
        return [stack] + out

    def shipped():
        pb = M._tuned_linear(z, w, **lin)
        pt = ttnn.permute(pb, (0, 3, 1, 2))
        ttnn.deallocate(pb)
        stack = M._sdpa_dense_mask(pt, am, HD, NK)
        ttnn.deallocate(pt)
        return blocks(stack)

    def pad2():
        pb = M._tuned_linear(z, w, **lin)
        pt = ttnn.permute(pb, (0, 3, 1, 2))
        ttnn.deallocate(pb)
        s = ttnn.add(ttnn.multiply(pt, HD ** 0.5), am)
        ttnn.deallocate(pt)
        stack = ttnn.pad(s, [(0, 0), (0, 0), (0, NK - I), (0, NK - I)], -1e4)
        ttnn.deallocate(s)
        return blocks(stack)

    def zpad_with(fold):
        def run():
            zp = ttnn.pad(z, [(0, 0), (0, NK - I), (0, NK - I), (0, 0)], 0.0)
            amp = pad_axis(pad_axis(am, NK, 3, -1e4), NK, 2, -1e4)
            pb = M._tuned_linear(zp, w, **lin)
            ttnn.deallocate(zp)
            pt = ttnn.permute(pb, (0, 3, 1, 2))
            ttnn.deallocate(pb)
            if fold:
                stack = ttnn.add(pt, amp, input_tensor_a_activations=[
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, HD ** 0.5)])
            else:
                sc = ttnn.multiply(pt, HD ** 0.5)
                stack = ttnn.add(sc, amp)
                ttnn.deallocate(sc)
            ttnn.deallocate(pt)
            ttnn.deallocate(amp)
            return blocks(stack)
        return run

    arms = {"shipped": shipped, "pad2": pad2, "zpad": zpad_with(False), "zpad_af": zpad_with(True)}
    rec = {"I": I, "n_key": NK, "heads": H, "reps": N, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "host": os.uname().nodename, "arch": str(dev.arch()), "arms": {}}
    base = None
    for name, fn in arms.items():
        try:
            o, med, lo, hi = timeit(fn, dev)
            full = ttnn.to_torch(o[0])
            assert tuple(full.shape) == (1, H, NK, NK), tuple(full.shape)
            d = full[:, :, :I, :I].double()
            assert torch.isfinite(d).all(), "non-finite output"
            blk = all(torch.equal(ttnn.to_torch(o[1 + i]), full[:, HB * i:HB * (i + 1)]) for i in (0, NB - 1))
            pads = bool((full[:, :, I:, :] == -1e4).all() and (full[:, :, :, I:] == -1e4).all())
            if base is None:
                base = full
            r = {"ms_median": med, "ms_min": lo, "ms_max": hi, "equal_shipped": bool(torch.equal(full, base)),
                 "pads_minus_1e4": pads, "blocks_equal_stack": blk,
                 "max_abs_f64": float((d - ref).abs().max()),
                 "rel_err_f64": float((d - ref).norm() / ref.norm())}
            for x in o:
                ttnn.deallocate(x)
        except Exception as e:                                           # noqa: BLE001
            r = {"error": repr(e)[:400]}
        rec["arms"][name] = r
        print(name, r, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

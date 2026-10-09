"""spd-bh: the pair Transition's row height on Blackhole at Protenix-v2's c=256.

    TT_VISIBLE_DEVICES=N python perf/spd_bh/transition_h.py --out OUT.jsonl [--shape 736x256x1024]
        [--hs 0,12,14,16,20,24] [--reps 7]

Blackhole raises the row height to its measured L1 budget only for c <= 128
(_BH_TRANSITION_L1_ROWS_MAX_C), so c=256 keeps the Wormhole-derived ratio: h=11 at 736 tokens
against an L1 budget of ~16 rows. This runs the real Transition module, both exits the fold uses
(plain and add_to_input), at the shipped height (h=0 = no override) and at forced heights via
TT_BIO_TRANSITION_H_CHUNK, and reports ms per call, AICLK, torch.equal against the shipped
height (swiglu is row-local, so any height should be bit-identical) and rel_rms against float64.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            out[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().split()[0])
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shape", default="736x256x1024")
    ap.add_argument("--hs", default="0,12,14,16,20,24")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--levers", default="normal", help="normal | fast | none: the set live in the run")
    a = ap.parse_args()
    log = open(a.out, "a")

    def emit(**kw):
        kw["t_unix"] = time.time()
        line = json.dumps(kw, default=str)
        log.write(line + "\n"); log.flush(); print(line, flush=True)

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    dev = T.get_device()
    names = {"normal": T.NORMAL_LEVERS, "fast": T.FAST_LEVERS, "none": frozenset()}[a.levers]
    with T.levers(names):
        run(a, emit, torch, ttnn, T, dev)
    emit(ev="end")


def run(a, emit, torch, ttnn, T, dev):
    kcls = (ttnn.WormholeComputeKernelConfig if T.is_wormhole()
            else ttnn.types.BlackholeComputeKernelConfig)
    fid = ttnn.MathFidelity.HiFi3 if T.lever("trunk_hifi3") else ttnn.MathFidelity.HiFi4
    ckc = kcls(math_fidelity=fid, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)
    S, C, HID = map(int, a.shape.split("x"))
    g = torch.Generator().manual_seed(0)
    bf = lambda t: t.to(torch.bfloat16).to(torch.float64)
    sd = {"norm.weight": bf(1 + 0.1 * torch.randn(C, generator=g)),
          "norm.bias": bf(0.1 * torch.randn(C, generator=g)),
          "fc1.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
          "fc2.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
          "fc3.weight": bf(torch.randn(C, HID, generator=g) / HID ** 0.5)}
    zt = bf(torch.randn(1, S, S, C, generator=g))
    F = torch.nn.functional
    xn = F.layer_norm(zt, (C,), sd["norm.weight"], sd["norm.bias"], 1e-5)
    ref = (F.silu(xn @ sd["fc1.weight"].T) * (xn @ sd["fc2.weight"].T)) @ sd["fc3.weight"].T
    del xn
    tr = T.Transition({k: v.float() for k, v in sd.items()}, ckc)
    z = ttnn.from_torch(zt.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    g_ = dev.compute_with_storage_grid_size()
    emit(ev="start", shape=a.shape, grid=[int(g_.x), int(g_.y)], arch=str(dev.arch()),
         chip=os.environ.get("TT_VISIBLE_DEVICES"), aiclk=aiclk(),
         l1_rows_max_c=getattr(T, "_BH_TRANSITION_L1_ROWS_MAX_C", None),
         levers=a.levers, fidelity=str(fid))

    def call(add):
        if add:
            zc = ttnn.clone(z)
            return tr(zc, add_to_input=True)
        return tr(z)

    base = {}
    for add in (False, True):
        for h in [int(v) for v in a.hs.split(",")]:
            if h:
                os.environ["TT_BIO_TRANSITION_H_CHUNK"] = str(h)
            else:
                os.environ.pop("TT_BIO_TRANSITION_H_CHUNK", None)
            T.TRANSITION_H_CHUNK_SHAPES.clear()
            arm = f"{'add' if add else 'plain'}_h{h or 'ship'}"
            try:
                for _ in range(2):
                    o = call(add); ttnn.synchronize_device(dev); ttnn.deallocate(o)
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter(); o = call(add); ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(o)
                clk = aiclk()
                o = call(add); out = ttnn.to_torch(o); ttnn.deallocate(o)
                want = ref + zt if add else ref
                d = out.to(torch.float64) - want
                heights = sorted({k[2] for k in T.TRANSITION_H_CHUNK_SHAPES})
                if h == 0:
                    base[add] = out
                emit(ev="arm", arm=arm, h_used=heights, ms=ts, ms_med=statistics.median(ts),
                     ms_min=min(ts), aiclk=clk,
                     equal_to_ship=bool(torch.equal(out, base[add])) if add in base else None,
                     rel_rms=float(d.pow(2).mean().sqrt() / want.pow(2).mean().sqrt()),
                     finite=bool(torch.isfinite(out).all()))
            except Exception as e:
                emit(ev="arm_fail", arm=arm, err=str(e)[:400])
    os.environ.pop("TT_BIO_TRANSITION_H_CHUNK", None)


if __name__ == "__main__":
    main()

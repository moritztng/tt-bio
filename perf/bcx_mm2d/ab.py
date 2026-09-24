#!/usr/bin/env python3
"""bcx-mm2d end to end: main's `tt_bio.autograd` against this branch's, on the census block.

Both modules are loaded into ONE process (main's from `git show <base>:tt_bio/autograd.py`), so
both arms share the card, the clock, the weights and the inputs. The block is `bcx-census`'s:
one pair unit (`_block` + the in-unit transition, c_z 128, 4 x 32 heads, chunk 128,
uncheckpointed) at n=256, and `b` is its device-side slope in the number of units, exactly as
the census defines it: median(fwd K2 - fwd K1) + median(bwd K2 - bwd K1) per arm, over pairs
whose arm order and K order both alternate.

Then, off the clock:
  * EXACT: one K=1 and one K=2 step per arm on identical inputs, recording every taped
    forward value and every gradient handed to `add_grad`, compared byte for byte.
  * float64: the K=1 forward logits and the logit gradient of each arm against
    `torch_chain` in float64 on the same bf16-rounded weights and logits, the gradient seeded
    with the float64 reference's own loss gradient.
"""
import argparse, importlib.util, json, pathlib, statistics, subprocess, sys, tempfile, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.hallgrad.e2e_distogram import ClockTrace, make_weights, torch_chain  # noqa: E402
from perf.hallgrad.census import (C_Z, C_S, HEADS, HEAD_DIM, BINS, CONTACT_BINS, MIN_SEP,  # noqa: E402
                                  one_step, stamp)


def load_base(ref):
    src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ref}:tt_bio/autograd.py"])
    path = pathlib.Path(tempfile.mkdtemp()) / "autograd_base.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("tt_bio._autograd_base", path,
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "tt_bio"
    spec.loader.exec_module(mod)
    return mod


class Capture:
    """Every taped forward value and every add_grad gradient, to host, in order."""

    def __init__(self, ag, ttnn):
        self.ag, self.ttnn, self.fwd, self.bwd = ag, ttnn, [], []
        self._tape, self._add = ag._tape, ag.Tensor.add_grad

    def __enter__(self):
        cap = self

        def tape(out_v, parents, make):
            cap.fwd.append(cap.ttnn.to_torch(out_v).clone())
            return cap._tape(out_v, parents, make)

        def add_grad(t, g):
            cap.bwd.append(cap.ttnn.to_torch(g).clone())
            return cap._add(t, g)
        self.ag._tape, self.ag.Tensor.add_grad = tape, add_grad
        return self

    def __exit__(self, *a):
        self.ag._tape, self.ag.Tensor.add_grad = self._tape, self._add


def same_bytes(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = torch.int16 if a.element_size() == 2 else torch.int32
    return bool(torch.equal(a.contiguous().view(iv), b.contiguous().view(iv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--no-f64", action="store_true")
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag_new
    ag_base = load_base(args.base)
    arms = {"base": ag_base, "mm2d": ag_new}

    device = tt.get_device()
    clocks = ClockTrace(period=1.0).start()
    blob = {"argv": " ".join(sys.argv), "stamp_start": stamp(), "n": args.n, "base": args.base,
            "base_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", args.base],
                                                text=True).strip()}

    def flush():
        blob["clock_meta"] = clocks.summary()
        blob["clock_samples"] = clocks.samples
        json.dump(blob, open(args.out, "w"), indent=1)

    n = args.n
    idx = np.arange(n)
    mask = np.abs(idx[:, None] - idx[None, :]) >= MIN_SEP
    M = int(mask.sum())
    mask_t = torch.from_numpy(mask).to(torch.float64)
    inC = torch.zeros(BINS, dtype=torch.float64)
    inC[:CONTACT_BINS] = 1.0
    logits = (np.random.default_rng(args.seed).standard_normal((n, 21)) * 0.1).astype(np.float32)

    Wnp, dev_w, cfgs = {}, {}, {}
    for K in (1, 2):
        Wnp[K] = make_weights(np.random.default_rng(args.seed), C_S, C_Z, HEADS, HEAD_DIM, C_Z, BINS,
                              blocks=K, block_transition=True)
        dev_w[K] = {k: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    for k, v in Wnp[K].items()}
        cfgs[K] = dict(heads=HEADS, head_dim=HEAD_DIM, hidden=C_Z, chunk=args.chunk,
                       checkpoint=False, blocks=K, block_transition=True)
    W = {(a, K): {k: arms[a].Tensor(v) for k, v in dev_w[K].items()} for a in arms for K in (1, 2)}

    def step(a, K):
        return one_step(arms[a], ttnn, device, W[(a, K)], cfgs[K], logits, mask_t, M, inC)

    print("# warm-up", flush=True)
    for _ in range(args.warm):
        for a in arms:
            for K in (1, 2):
                step(a, K)

    print(f"# {args.pairs} rounds, arm order and K order alternating", flush=True)
    rec = {a: {1: [], 2: []} for a in arms}
    t0 = time.time()
    for i in range(args.pairs):
        order = ["base", "mm2d"] if i % 2 == 0 else ["mm2d", "base"]
        ks = (1, 2) if (i // 2) % 2 == 0 else (2, 1)
        for a in order:
            for K in ks:
                rec[a][K].append(step(a, K))
    t1 = time.time()
    blob["timed"] = {"window": [t0, t1], "clock_window": clocks.window(t0, t1)}
    for a in arms:
        r1, r2 = rec[a][1], rec[a][2]
        f = [y["fwd_s"] - x["fwd_s"] for x, y in zip(r1, r2)]
        bw = [y["bwd_s"] - x["bwd_s"] for x, y in zip(r1, r2)]
        st = [y["step_s"] - x["step_s"] for x, y in zip(r1, r2)]
        blob["timed"][a] = {
            "b_device_side": statistics.median(f) + statistics.median(bw),
            "fwd_slope_median": statistics.median(f), "bwd_slope_median": statistics.median(bw),
            "b_step_paired": st, "b_step_median": statistics.median(st),
            "b_step_p10": float(np.percentile(st, 10)), "b_step_p90": float(np.percentile(st, 90)),
            "fwd_slope_p10": float(np.percentile(f, 10)), "fwd_slope_p90": float(np.percentile(f, 90)),
            "bwd_slope_p10": float(np.percentile(bw, 10)), "bwd_slope_p90": float(np.percentile(bw, 90)),
            "t_K1": [r["step_s"] for r in r1], "t_K2": [r["step_s"] for r in r2],
            "fwd_K1": [r["fwd_s"] for r in r1], "fwd_K2": [r["fwd_s"] for r in r2],
            "bwd_K1": [r["bwd_s"] for r in r1], "bwd_K2": [r["bwd_s"] for r in r2],
        }
        tb = blob["timed"][a]
        print(f"  {a:5s} b {tb['b_device_side']:.5f} s (fwd {tb['fwd_slope_median']:.5f} "
              f"[{tb['fwd_slope_p10']:.5f},{tb['fwd_slope_p90']:.5f}] + bwd {tb['bwd_slope_median']:.5f} "
              f"[{tb['bwd_slope_p10']:.5f},{tb['bwd_slope_p90']:.5f}])  step-paired median "
              f"{tb['b_step_median']:.5f} p10 {tb['b_step_p10']:.5f} p90 {tb['b_step_p90']:.5f}", flush=True)
    print(f"  AICLK {blob['timed']['clock_window']}", flush=True)
    blob["timed"]["speedup_b"] = blob["timed"]["base"]["b_device_side"] / blob["timed"]["mm2d"]["b_device_side"]
    print(f"  b speedup {blob['timed']['speedup_b']:.3f}x", flush=True)
    flush()

    print("# exactness, every taped value and every gradient", flush=True)
    blob["exact"] = {}
    outs = {}
    for K in (1, 2):
        caps = {}
        for a in arms:
            with Capture(arms[a], ttnn) as c:
                outs[(a, K)] = one_step_keep(arms[a], ttnn, device, W[(a, K)], cfgs[K], logits,
                                             mask_t, M, inC)
            caps[a] = c
        fb, fm = caps["base"].fwd, caps["mm2d"].fwd
        gb, gm = caps["base"].bwd, caps["mm2d"].bwd
        fdiff = [i for i, (x, y) in enumerate(zip(fb, fm)) if not same_bytes(x, y)]
        gdiff = [i for i, (x, y) in enumerate(zip(gb, gm)) if not same_bytes(x, y)]
        e = {"fwd_values": len(fb), "fwd_values_mm2d": len(fm), "fwd_differing": len(fdiff),
             "bwd_grads": len(gb), "bwd_grads_mm2d": len(gm), "bwd_differing": len(gdiff),
             "first_fwd_diff": fdiff[:5], "first_bwd_diff": gdiff[:5],
             "logits_out_identical": same_bytes(outs[("base", K)]["d"], outs[("mm2d", K)]["d"]),
             "logit_grad_identical": same_bytes(outs[("base", K)]["g"], outs[("mm2d", K)]["g"])}
        if fdiff:
            x, y = fb[fdiff[0]].double(), fm[fdiff[0]].double()
            e["first_fwd_diff_rel_l2_between_arms"] = float((x - y).norm() / x.norm())
        blob["exact"][K] = e
        print(f"  K={K}: fwd {len(fb)}/{len(fm)} values, {len(fdiff)} differ; bwd {len(gb)}/{len(gm)} "
              f"grads, {len(gdiff)} differ; logits identical {e['logits_out_identical']}, "
              f"logit grad identical {e['logit_grad_identical']}", flush=True)
    flush()

    if not args.no_f64:
        print("# float64 reference, K=1 and K=2", flush=True)
        blob["f64"] = {}
        bf = lambda v: torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16).to(torch.float64)
        for K in (1, 2):
            W64 = {k: bf(v) for k, v in Wnp[K].items()}
            l64 = bf(logits).requires_grad_(True)
            d64 = torch_chain(l64, W64, cfgs[K])
            p = torch.softmax(d64.detach(), dim=-1)
            pc = p[..., :CONTACT_BINS].sum(-1).clamp_min(1e-12)
            seed = -(p * (inC / pc.unsqueeze(-1) - 1.0)) * (mask_t / M).unsqueeze(-1)
            d64.backward(seed)
            g64 = l64.grad.detach()
            r = {}
            for a in arms:
                d, g = outs[(a, K)]["d"].double(), outs[(a, K)]["g"].double()
                r[a] = {"logits_rel_l2": float((d - d64.detach()).norm() / d64.detach().norm()),
                        "logits_max_abs": float((d - d64.detach()).abs().max()),
                        "grad_rel_l2": float((g - g64).norm() / g64.norm()),
                        "grad_cos": float((g * g64).sum() / (g.norm() * g64.norm()))}
                print(f"  K={K} {a:5s} logits rel_l2 {r[a]['logits_rel_l2']:.4e} max_abs "
                      f"{r[a]['logits_max_abs']:.4e} | logit grad rel_l2 {r[a]['grad_rel_l2']:.4e} "
                      f"cos {r[a]['grad_cos']:.7f}", flush=True)
            blob["f64"][K] = r
            flush()

    clocks.stop()
    blob["stamp_end"] = stamp()
    flush()
    print(f"# wrote {args.out}")


def one_step_keep(ag, ttnn, device, Wtt, cfg, logits_np, mask_t, M, inC):
    """census.one_step, keeping the distogram logits and the logit gradient on host."""
    from perf.hallgrad.e2e_distogram import tt_chain
    lt = ag.Tensor(ttnn.from_torch(torch.from_numpy(logits_np).to(torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                   requires_grad=True)
    d = tt_chain(ag, ttnn, lt, Wtt, cfg)
    dh = ttnn.to_torch(d.value).clone()
    z64 = dh.to(torch.float64)
    p = torch.softmax(z64, dim=-1)
    pc = p[..., :CONTACT_BINS].sum(-1).clamp_min(1e-12)
    seed_t = -(p * (inC / pc.unsqueeze(-1) - 1.0)) * (mask_t / M).unsqueeze(-1)
    d.backward(seed=ttnn.from_torch(seed_t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=device))
    g = ttnn.to_torch(lt.grad).clone()
    lt.grad = lt.node = d.grad = d.node = None
    return {"d": dh, "g": g}


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The customer-chart reproduction: Protenix-v2 at the customer's protocol, on a stock p150a.

A customer benchmarked Protenix-v2 on a p150a at 5 diffusion samples / 200 steps / 10 recycles
against a 580 aa target with 64-180 aa binders and reported a 208 s median. This measures the
same protocol at three token counts that land in three different shipped token buckets, so the
result tests the LEVEL and the SHAPE of their panel B at once.

One process, one device open. Each (size, n_sample) pair needs its own ``build_fold`` because the
sample count is baked into the job config, so the plan is a list of legs and each leg folds cold
once (discarded) then warm n times. Results are written after EVERY fold: a turn that runs out of
time still lands what it measured.

The clock is sampled DURING every fold (`perf/clocksample.py`). A fold seconds figure without the
AICLK it ran at is not a measurement on this part.
"""
import argparse, hashlib, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    # leg = SIZE:NSAMPLE:NWARM   e.g. 652:5:3
    ap.add_argument("--legs", required=True, help="comma-separated SIZE:NSAMPLE:NWARM")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn                                                    # noqa: F401
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import clocksample
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "commit": os.environ.get("PVXC_COMMIT"), "clock_pin": os.environ.get("PVXC_PIN"),
           "legs": a.legs, "runs": []}

    legs = []
    for tok in a.legs.split(","):
        sz, ns, nw = (int(x) for x in tok.split(":"))
        legs.append((sz, ns, nw))

    dev_opened = False
    for size, nsample, nwarm in legs:
        tgt = a.fixdir / f"cdk2x2_{size}.yaml"
        a3m = a.fixdir / f"cdk2x2_{size}.a3m"
        assert tgt.exists() and a3m.exists(), f"missing fixture for {size}"
        fix = {"yaml_sha256": hashlib.sha256(tgt.read_bytes()).hexdigest(),
               "a3m_sha256": hashlib.sha256(a3m.read_bytes()).hexdigest()}
        t0 = time.perf_counter()
        one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_pvxc_{size}", tgt, a3m,
                                             samples=nsample)
        load_s = time.perf_counter() - t0
        if not dev_opened:
            DEV = T.get_device()
            g = DEV.compute_with_storage_grid_size()
            res["grid"] = [g.x, g.y]
            dev_opened = True
        print(f"[leg] size={size} nsample={nsample} nwarm={nwarm} load={load_s:.1f}s "
              f"grid={res['grid']}", flush=True)

        for i in range(nwarm + 1):
            tag = "cold" if i == 0 else "warm"
            with clocksample.during(period=2.0) as clk:
                tw = time.perf_counter()
                fold_s, m = one_fold()
                wall = time.perf_counter() - tw
            # struct_dir is cleared at the START of the next fold, so hash it now or never.
            # With n_sample=5 predict_one writes five CIFs; the count is itself the proof that
            # five samples were produced rather than one.
            sd = Path(meta["struct_dir"])
            cifs = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                    for f in sorted(sd.glob("*.cif"))}
            assert m.get("msa"), "fold ran without an MSA -- cache seeding failed"
            rec = {"size": size, "n_sample": nsample, "ix": i, "tag": tag,
                   "n_cif": len(cifs), "cifs": cifs,
                   "fold_s": round(fold_s, 4), "wall_s": round(wall, 4),
                   "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
                   "load_s": round(load_s, 2), "n_msa": meta.get("n_msa"),
                   "loadavg1": round(os.getloadavg()[0], 2),
                   "clock": clk.summary(), "clock_line": clk.line(0), **fix}
            res["runs"].append(rec)
            print(f"  [{tag}] size={size} ns={nsample} fold {fold_s:.3f}s "
                  f"n_tokens={m.get('n_tokens')} n_cif={len(cifs)} plddt={m.get('plddt')} "
                  f"load={rec['loadavg1']} | {clk.line(0)}", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))

    # summary per leg
    summ = []
    for size, nsample, _ in legs:
        w = [r["fold_s"] for r in res["runs"]
             if r["size"] == size and r["n_sample"] == nsample and r["tag"] == "warm"]
        if not w:
            continue
        summ.append({"size": size, "n_sample": nsample, "n_warm": len(w),
                     "median_s": round(st.median(w), 3), "min_s": round(min(w), 3),
                     "max_s": round(max(w), 3),
                     "spread_s": round(max(w) - min(w), 3)})
    res["summary"] = summ
    a.out.write_text(json.dumps(res, indent=1))
    for s in summ:
        print(f"SUMMARY size={s['size']} ns={s['n_sample']} n={s['n_warm']} "
              f"median={s['median_s']}s spread={s['spread_s']}s", flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

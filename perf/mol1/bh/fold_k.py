"""Mol-1 folds of one input, K seeds at once (`Protenix.fold_k`), against the same seeds folded one at a time.

mol1-bh-batch. The model and the input come through the worker exactly as perf/mol1/fair/run.py builds them (the
`tt-bio predict` config, `_WorkerState.load_model`, `_protenix_inputs`), then the fold calls are timed directly:
  solo     fold(seed) for the first two seeds, warm, and each solo's coordinates
  samples  fold(seed, n_sample=10 / 20): what K=2 / 4 members' diffusion would cost as one trajectory
  fold_k   K = --ks seeds through one call, each member's coordinates against its solo fold
AICLK sampled during. Not the gate's span (no output writing): an A/B of the engine call only.

Run (cwd = the inputs dir): TT_VISIBLE_DEVICES=<card> python perf/mol1/bh/fold_k.py --input mol1-256.json --out d
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parents[1] / "fair")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,4")
    ap.add_argument("--samples", default="10,20")
    ap.add_argument("--extra", default="pair_batch", help="levers added to the model's set")
    ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from clocksample import during
    from run import run_config
    from tt_bio import worker as W
    import tt_bio.tenstorrent as T

    cfg, jobs = run_config("mol1", Path(a.input).resolve(), out)
    W._ensure_local_artifacts(cfg)
    state = W._WorkerState("tenstorrent")
    state.load_model(cfg)
    m = state.model
    m._levers = m._levers | T.parse_levers([x for x in a.extra.split(",") if x])
    feats, _chains, _specs = state._protenix_inputs(jobs[0], cfg)
    kw = dict(n_step=cfg["sampling_steps"], n_sample=int(cfg["diffusion_samples"]), return_confidence=True,
              n_cycles=cfg.get("recycling_steps"), max_parallel_samples=cfg.get("max_parallel_samples"),
              trace=W._protenix_trace(cfg) and W._protenix_region_reserved(), return_logits=True)
    res = {"input": a.input, "extra": a.extra, "kw": {k: v for k, v in kw.items()}, "levers": sorted(m._levers),
           "tokens": int(feats["restype"].shape[0]) if "restype" in feats else None, "solo": {}, "samples": {}, "k": {}}

    def save():
        (out / "fold_k.json").write_text(json.dumps(res, indent=1, default=str))

    def timed(fn):
        ttnn.synchronize_device(m.dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(m.dev)
        return time.perf_counter() - t0, r

    def coords(r):
        return (r[0] if isinstance(r, tuple) else r).detach().float().cpu()

    def cmp(x, y):
        d = (x.double() - y.double())
        return {"equal": bool(torch.equal(x, y)), "max_abs_A": d.abs().max().item(),
                "rms_A": d.pow(2).sum(-1).mean().sqrt().item()}

    ks = [int(k) for k in a.ks.split(",") if k]
    seeds = [a.seed + i for i in range(max(ks + [2]))]
    with torch.no_grad(), during() as clk:
        res["cold_s"], _ = timed(lambda: m.fold(feats, seed=seeds[0], **kw))
        save()
        solo = {}
        for i, sd in enumerate(seeds[:2]):
            t, r = timed(lambda: m.fold(feats, seed=sd, **kw))
            solo[sd] = coords(r)
            res["solo"][str(sd)] = {"s": t}
            if i == 0:
                t2, r2 = timed(lambda: m.fold(feats, seed=sd, **kw))
                res["solo"][str(sd)]["s_repeat"] = t2
                res["solo"][str(sd)]["repeat_vs_first"] = cmp(coords(r2), solo[sd])
            save()
        for n in [int(x) for x in a.samples.split(",") if x]:
            t, _ = timed(lambda: m.fold(feats, seed=seeds[0], **dict(kw, n_sample=n)))
            res["samples"][str(n)] = {"s": t}
            save()
        for k in ks:
            try:
                rec = res["k"][str(k)] = {}
                for rep in range(2):
                    t, rs = timed(lambda: m.fold_k(feats, seeds[:k], **kw))
                    rec.setdefault("s", []).append(t)
                rec["s_per_pred"] = min(rec["s"]) / k
                rec["members_vs_solo"] = {str(sd): cmp(coords(rs[i]), solo[sd]) for i, sd in enumerate(seeds[:k])
                                          if sd in solo}
            except Exception as e:  # noqa: BLE001 -- a K that raises is a result
                import traceback
                res["k"][str(k)] = {"error": f"{type(e).__name__}: {str(e)[:500]}", "tb": traceback.format_exc()[-4000:]}
            save()
    res["aiclk"] = clk.summary()
    save()


if __name__ == "__main__":
    main()

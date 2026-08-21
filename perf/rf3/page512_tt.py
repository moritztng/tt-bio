#!/usr/bin/env python3
"""RF3's cell for the 512 aa perf page: the page's own fixture, protocol and metric.

The page's cells are whole warm folds (`s_per_fold`) of `perf/size512/fixtures/cdk2x2_512.yaml`
with its fixed 35-sequence MSA, one prediction at a time on one accelerator, at each model's own
shipped recycle and sampling-step settings. RF3's existing ladder in `perf/rf3/` measures device
phases only, on a single-sequence input, so none of it can go on the page; this harness produces
what can.

Deliberately not `perf/rf3/tt_rf3_bench.py`: that one times the trunk and diffusion in isolation
with a feature cache. Here the timed region is `predict_one` -- host featurisation, fold and CIF
write -- which is the boundary every other cell on the page was measured at.

Three things are asserted rather than assumed, because each has silently produced a wrong number
before: that `tt_bio` resolves inside this checkout and not the installed wheel, that the
executed denoise-step count matches the requested schedule, and that every fold (cold included)
actually read the alignment.
"""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=4, help="warm folds after the discarded cold one")
    ap.add_argument("--label", default="shipped")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    from tt_bio.rf3.sampler import DiffusionSampler

    assert Path(T.__file__).resolve().is_relative_to(ROOT), \
        f"tt_bio resolves to {T.__file__}, not this checkout -- set PYTHONPATH"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "rf3")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "rf3")
    # Upstream's shipped inference engine (foundry models/rf3/configs/inference_engine/rf3.yaml)
    # ships n_recycles: 10 and num_steps: 50. Publishing anything else would put RF3 on the page
    # at a configuration it does not ship, so this fails loudly instead of measuring quietly.
    assert (B.RECYCLING_STEPS, B.SAMPLING_STEPS) == (10, 50), \
        f"rf3 defaults are {B.RECYCLING_STEPS}/{B.SAMPLING_STEPS}, expected 10/50"

    # The schedule has num_timesteps entries and the rollout consumes consecutive pairs, so a
    # 50-step schedule executes 49 denoise calls. Count them per fold: the config value proves
    # what was requested, only the counter proves what ran.
    steps = {"n": 0}
    _orig_sample = DiffusionSampler.sample

    def counting_sample(self, denoise, *args, **kw):
        def counted(*a, **k):
            steps["n"] += 1
            return denoise(*a, **k)
        return _orig_sample(self, counted, *args, **kw)

    DiffusionSampler.sample = counting_sample

    tgt = a.fixdir / "cdk2x2_512.yaml"
    a3m = a.fixdir / "cdk2x2_512.a3m"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    one_fold, meta, state = B.build_fold("rf3", ROOT / ".msa_rf3_page512", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as im
    res = {"label": a.label, "model": "rf3", "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "ttnn": im.version("ttnn"),
           "hardware": meta.get("hardware"), "card_type": meta.get("card_type"),
           "aiclk_mhz": meta.get("aiclk_mhz"), "load_s": meta.get("load_s"),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "diffusion_samples": B.DIFFUSION_SAMPLES, "seed": B.SEED,
           "n_msa": meta.get("n_msa"), "timed_region": meta.get("timed_region"),
           "target": str(tgt), "a3m": str(a3m),
           "sha256_target": sha(tgt), "sha256_a3m": sha(a3m),
           "env_flags": {k: v for k, v in sorted(os.environ.items())
                         if k.startswith("TT_BIO_")},
           "folds": []}

    def one(tag):
        steps["n"] = 0
        fold_s, m = one_fold()
        assert m.get("msa"), f"{tag}: fold ran without an MSA -- cache seeding failed"
        assert m.get("n_tokens") == 512, f"{tag}: n_tokens {m.get('n_tokens')}, expected 512"
        assert steps["n"] == B.SAMPLING_STEPS - 1, \
            f"{tag}: executed {steps['n']} denoise steps, expected {B.SAMPLING_STEPS - 1}"
        return {"tag": tag, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
                "ptm": m.get("ptm"), "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
                "denoise_calls": steps["n"], "msa": m.get("msa"),
                "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                               for p in sorted(struct_dir.glob("*.cif"))},
                "loadavg": open("/proc/loadavg").read().split()[:3],
                "opm_small_depth_stats": list(T.OPM_SMALL_DEPTH_STATS),
                "triatt_fused_hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS)}

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")

    res["cold"] = one("cold")
    print(f"[{a.label}] cold {res['cold']['fold_s']:.2f}s plddt={res['cold']['plddt']}", flush=True)
    flush()
    for i in range(a.repeat):
        rec = one(f"warm{i}")
        res["folds"].append(rec)
        print(f"[{a.label}] warm{i} {rec['fold_s']:.3f}s plddt={rec['plddt']} "
              f"cif={list(rec['cif_sha256'].values())}", flush=True)
        flush()

    w = sorted(f["fold_s"] for f in res["folds"])
    res["warm_walls_s"] = w
    res["median_s"] = round(statistics.median(w), 3)
    res["min_s"], res["max_s"] = w[0], w[-1]
    res["spread_s"] = round(w[-1] - w[0], 3)
    res["spread_pct"] = round(100.0 * (w[-1] - w[0]) / res["median_s"], 2)
    digests = {tuple(sorted(f["cif_sha256"].values())) for f in res["folds"]}
    res["warm_digest_identical"] = len(digests) == 1
    res["plddts"] = sorted({f["plddt"] for f in res["folds"]})
    flush()
    print(f"[{a.label}] median {res['median_s']:.3f}s spread {res['spread_s']:.3f}s "
          f"({res['spread_pct']}%) digest_identical={res['warm_digest_identical']} "
          f"plddts={res['plddts']}", flush=True)
    print(f"[{a.label}] OPM_SMALL_DEPTH_STATS {res['folds'][-1]['opm_small_depth_stats']} "
          f"TRIATT {res['folds'][-1]['triatt_fused_hifi_stats']}", flush=True)
    print("wrote", a.out, flush=True)
    state.reset()
    T.cleanup()


if __name__ == "__main__":
    main()

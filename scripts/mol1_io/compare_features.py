"""tt-bio's Protenix-family featurizer against Mol-1's own, key by key, on the goldens in tests/data/mol1_io/features.

    python scripts/mol1_io/compare_features.py [no_msa|msa|rna_msa] [job ...]

Prints one line per feature: equal, or shape / max abs difference / count of differing entries. The test
(tests/test_mol1_features.py) imports `tt_features` and `diff` from here.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
GOLD = REPO / "tests" / "data" / "mol1_io" / "features"
CASES = REPO / "tests" / "data" / "mol1_io" / "synthetic.json"
MODES = {"no_msa": dict(single_sequence=True), "msa": {}, "rna_msa": dict(use_rna_msa=True)}


def tt_features(job: dict, mode: str, tmp: Path) -> dict:
    """Stage an upstream JSON job the way `tt-bio predict jobs.json` does and run the worker's featurizer."""
    from tt_bio import protenix_json
    from tt_bio.worker import _WorkerState

    tmp.mkdir(parents=True, exist_ok=True)
    src = tmp / f"{job['name']}.json"
    src.write_text(json.dumps([job]))
    staged = protenix_json.stage(src, tmp / "staged")
    yaml = sorted(Path(staged).glob(f"{job['name']}*.yaml"))[0]
    state = object.__new__(_WorkerState)
    state.model, state.accelerator = None, "cpu"
    cfg = {"model": "mol1", "msa_dir": str(tmp / "msa"), "template_structures": str(tmp / "tmpl"),
           "output_format": "cif", **MODES[mode]}
    feats, _chains, _specs = state._protenix_inputs(yaml, cfg)
    return {k: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v) for k, v in feats.items()
            if hasattr(v, "shape")}


def diff(ours: dict, gold: dict) -> dict:
    """key -> None when equal, else a short description. Integer/bool features compare exactly."""
    out = {}
    for k in sorted(set(gold) | set(ours)):
        if k.startswith("atoms."):
            continue
        if k not in ours or k not in gold:
            out[k] = "missing in tt-bio" if k not in ours else "tt-bio only"
            continue
        a, g = np.asarray(ours[k]), np.asarray(gold[k])
        a = a.reshape(g.shape) if a.size == g.size and a.shape != g.shape and a.squeeze().shape == g.squeeze().shape else a
        if a.shape != g.shape:
            out[k] = f"shape {a.shape} vs upstream {g.shape}"
            continue
        d = np.abs(a.astype(np.float64) - g.astype(np.float64))
        n = int((d > 0).sum())
        out[k] = None if n == 0 else f"{n}/{d.size} differ, max {d.max():.3g}"
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "no_msa"
    want = set(sys.argv[2:])
    jobs = {j["name"]: j for j in json.loads(CASES.read_text())}
    import tempfile
    for npz in sorted((GOLD / mode).glob("*.npz")):
        name = npz.stem
        if (want and name not in want) or name not in jobs:
            continue
        gold = dict(np.load(npz))
        with tempfile.TemporaryDirectory() as t:
            try:
                ours = tt_features(jobs[name], mode, Path(t))
            except Exception as e:  # report, keep going
                print(f"== {name}: tt-bio featurizer raised {type(e).__name__}: {e}")
                continue
        res = diff(ours, gold)
        bad = {k: v for k, v in res.items() if v}
        print(f"== {name}: {len(res) - len(bad)} equal, {len(bad)} differ")
        for k, v in bad.items():
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()

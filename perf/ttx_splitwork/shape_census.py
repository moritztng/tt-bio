#!/usr/bin/env python3
"""Which shapes the channel move is actually asked for in a real fold, and how often.

The 512 aa fold A/B came back inside its own A/A floor, so the question is no longer "is the op
faster" -- perf/ttx_splitwork/assign_ab.json settles that -- but "how much of a fold is this op".
That needs the shapes the fold really passes, not the ones a sweep picked. Records every call to
all three legs, accepted or refused, and prints the census.
"""
from __future__ import annotations

import argparse, collections, importlib.util, json, shutil, socket, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)
FIX = REPO / "perf" / "size512" / "fixtures"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixtures", default="cdk2x2_512")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    args = ap.parse_args()
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import tt_bio.reblock_permute as RB
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    seen = collections.Counter()

    def wrap(name, fn, taken):
        def inner(x, memory_config=None, device=None):
            seen[(name, tuple(int(v) for v in x.shape), taken(x, memory_config))] += 1
            return fn(x, memory_config, device)
        return inner

    RB.reblock_permute = wrap("fwd", RB.reblock_permute, RB.eligible)
    RB.reblock_permute_back = wrap("back", RB.reblock_permute_back, RB.eligible_back)
    import tt_bio.tenstorrent as TT
    TT._reblock.reblock_permute = RB.reblock_permute
    TT._reblock.reblock_permute_back = RB.reblock_permute_back
    _g = RB.reblock_permute_gated

    def gated(x, *a, **k):
        seen[("gated", tuple(int(v) for v in x.shape), True)] += 1
        return _g(x, *a, **k)
    RB.reblock_permute_gated = gated
    TT._reblock.reblock_permute_gated = gated

    allrows = []
    for fixture in args.fixtures.split(","):
      work = Path(tempfile.mkdtemp(prefix="ttx-census-"))
      struct_dir = work / "out"; struct_dir.mkdir(parents=True)
      msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
      LEV._seed_msa(FIX / f"{fixture}.yaml", (FIX / f"{fixture}.a3m").read_text(), msa_dir)
      cfg = LEV.build_cfg(msa_dir, struct_dir)
      LEV._ensure_local_artifacts = _ensure_local_artifacts
      _ensure_local_artifacts(cfg)
      state = _WorkerState("tenstorrent")
      state.load_model(cfg)
      state.bind_run("ttx-reblock-cores-ship-census", cfg)

      seen.clear()
      t0 = time.perf_counter()
      state.predict_one(FIX / f"{fixture}.yaml", cfg)
      wall = time.perf_counter() - t0

      rows = [{"leg": k[0], "shape": list(k[1]), "eligible": bool(k[2]), "calls": n}
              for k, n in sorted(seen.items(), key=lambda kv: -kv[1])]
      allrows.append({"fixture": fixture, "fold_s": round(wall, 3), "rows": rows})
      print(f"{fixture}: fold {wall:.3f}s")
      for r in rows:
          print(f"  {r['leg']:>6} {str(r['shape']):>24} eligible={r['eligible']!s:5} "
                f"calls={r['calls']}")
      args.out.parent.mkdir(parents=True, exist_ok=True)
      args.out.write_text(json.dumps(
          {"host": socket.gethostname(), "steps": args.steps, "recycles": args.recycles,
           "folds": allrows}, indent=1))
      shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

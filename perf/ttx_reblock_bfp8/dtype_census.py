#!/usr/bin/env python3
"""What DTYPE does the engine actually hand the channel-move gates? The question behind the bfp8 ask.

The bfp8 proposal reads two completely different ways and they have opposite conclusions, so which
one is live has to be measured before anything is built:

  (a) something already hands this op a bfloat8_b tensor, the gate declines it on
      `*_dtype_layout`, and those calls silently fall back to `ttnn.permute`. Then a bfp8 kernel
      costs NO accuracy at all -- the precision was already spent upstream -- and the win is
      whatever the custom kernel beats the stock permute by.
  (b) every caller is bf16. Then "accept bfp8" means casting on the way in, which is a real
      precision cut AND two extra whole-tensor passes, and it has to earn both.

`tt_bio.tenstorrent._dtype()` returns bfloat8_b under `--fast`, which is what makes (a) plausible,
so both modes are folded here. The instrument is the gates themselves: each of the three is wrapped
so every tensor presented to it is counted by (dtype, layout, shape) whatever the gate then decides.
A dtype that never appears in this census has no traffic, and a lever with no traffic is worth zero
however fast its kernel is.

Short folds on purpose. This counts calls and reads dtypes; it times nothing, so the step and
recycle counts are set low enough to keep the census cheap and the numbers here must not be quoted
as fold times.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:ttx-reblock-bfp8-pairtrack \
        /home/ttuser/tt-bio-dev/env/bin/python3 perf/ttx_reblock_bfp8/dtype_census.py \
            --out perf/ttx_reblock_bfp8/dtype_census_qb2c3.json
"""
from __future__ import annotations

import argparse, collections, importlib.util, json, os, shutil, socket, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)

SEEN: dict = collections.defaultdict(int)


def install_census(rp):
    """Wrap the three gates so every presented tensor is counted, verdict included.

    The wrapper calls through, so the fold behaves exactly as it would without the census and the
    verdict is recorded rather than forced. `eligible_gated` takes the wide tensor first and a
    `slice_c` after it, so the tensor is positional-0 in all three.
    """
    for name in ("eligible", "eligible_back", "eligible_gated"):
        fn = getattr(rp, name)

        def wrapped(*a, _fn=fn, _name=name, **k):
            t = a[0]
            verdict = _fn(*a, **k)
            SEEN[(_name, str(t.dtype), str(t.layout), tuple(int(d) for d in t.shape),
                  bool(verdict))] += 1
            return verdict

        setattr(rp, name, wrapped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--fixture", default="cdk2x2_298")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    import tt_bio.reblock_permute as rp
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    install_census(rp)
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    fixdir = REPO / "perf" / "size512" / "fixtures"
    yaml_src = fixdir / f"{args.fixture}.yaml"
    a3m_src = fixdir / f"{args.fixture}.a3m"
    assert yaml_src.exists(), yaml_src

    tmp = Path(tempfile.mkdtemp(prefix="rbcensus_"))
    msa_dir, struct_dir = tmp / "msa", tmp / "struct"
    msa_dir.mkdir(); struct_dir.mkdir()
    target = tmp / yaml_src.name
    shutil.copy2(yaml_src, target)
    LEV._seed_msa(target, a3m_src.read_text(), msa_dir)

    dev = get_device()
    g = dev.compute_with_storage_grid_size()

    out = {
        "host": socket.gethostname(),
        "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": f"{g.x}x{g.y}",
        "fixture": args.fixture, "steps": args.steps, "recycles": args.recycles,
        "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "arms": {},
    }

    for fast in (False, True):
        SEEN.clear()
        cfg = LEV.build_cfg(msa_dir, struct_dir)
        cfg["fast"] = fast
        _ensure_local_artifacts(cfg)
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        # A fresh state per arm: --fast changes the stored weight dtypes, so an arm must build
        # its own model rather than inherit the other arm's caches.
        state = _WorkerState("tenstorrent")
        state.load_model(cfg)
        state.bind_run("ttx-reblock-bfp8", cfg)
        state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)

        rows = []
        by_dtype: dict = collections.Counter()
        for (gate, dt, lay, shape, verdict), n in sorted(SEEN.items(), key=lambda kv: -kv[1]):
            rows.append({"gate": gate, "dtype": dt, "layout": lay, "shape": list(shape),
                         "eligible": verdict, "calls": n})
            by_dtype[dt] += n
        out["arms"][f"fast={fast}"] = {
            "fast_mode": bool(TT._FAST_MODE),
            "trimul_chunk_dtype_helper": str(TT._dtype()),
            "total_gate_calls": sum(by_dtype.values()),
            "calls_by_dtype": dict(by_dtype),
            "rows": rows,
        }
        print(f"\n=== fast={fast}  _dtype()={TT._dtype()}  gate calls {sum(by_dtype.values())}")
        print("    by dtype:", dict(by_dtype))
        for r in rows[:12]:
            print(f"    {r['calls']:7d}  {r['gate']:16s} {r['dtype']:18s} "
                  f"{str(r['shape']):22s} eligible={r['eligible']}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shape census + in-fold per-site accuracy for the tuned `in0_block_w` program config.

One real fold, with `tt_bio.tenstorrent._linear` replaced by a spy that records
(caller, mt, kt, nt, fired?) for every call. With --rmsd the spy additionally runs the
untuned `core_grid` call on the SAME live operands the first time it sees a firing site and
reports max|delta| and rmsd of one against the other.

Both halves have to be in-fold: an op-level sweep passed a ttnn config that returned wrong
results only inside a real fold (perf/ktiles/infold_diff.py), and a guard written against a
logical shape has silently never fired in a real fold twice in this codebase.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--a3m", type=Path,
                    default=ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rmsd", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    B.RECYCLING_STEPS = a.recycles
    B.SAMPLING_STEPS = a.steps

    orig_linear = ttnn.linear
    counts: collections.Counter = collections.Counter()
    acc: dict = {}

    def spy(x, w, **kw):
        pc = T._tuned_config_for(x, w)
        xp, wp = tuple(x.padded_shape), tuple(w.padded_shape)
        m = 1
        for d in xp[:-1]:
            m *= int(d)
        f = sys._getframe(1)
        key = (f.f_globals.get("__name__", "?").split(".")[-1], f.f_code.co_name,
               f.f_lineno, m // 32, int(xp[-1]) // 32, int(wp[-1]) // 32,
               pc is not None)
        counts[key] += 1
        if pc is None:
            return orig_linear(x, w, **kw)
        kw_t = {k: v for k, v in kw.items() if k != "core_grid"}
        try:
            out = orig_linear(x, w, program_config=pc, **kw_t)
        except RuntimeError as e:      # same in-fold L1 recovery _linear does
            if "circular buffer" not in str(e):
                raise
            T._TUNED_L1_REJECTED.add((tuple(x.padded_shape), tuple(w.padded_shape)))
            counts[key] -= 1
            counts[key[:-1] + (False,)] += 1
            return orig_linear(x, w, **kw)
        if a.rmsd and key not in acc:
            base = orig_linear(x, w, **kw)
            t = ttnn.to_torch(out).float()
            b = ttnn.to_torch(base).float()
            ttnn.deallocate(base)
            d = (t - b).abs()
            scale = b.abs().max().item()
            acc[key] = dict(
                max_abs=d.max().item(), rmsd=((t - b) ** 2).mean().sqrt().item(),
                ref_absmax=scale, rms_ref=(b ** 2).mean().sqrt().item(),
                rel_max=(d.max().item() / scale if scale else 0.0),
                in0_block_w=int(pc.in0_block_w),
                cfg=type(pc).__name__.replace("Matmul", "").replace("ProgramConfig", ""),
            )
        return out

    # protenix.py does `from .tenstorrent import _linear`, so its 12 sites hold their own
    # module-level binding: patching T._linear alone would silently miss every one of them.
    import tt_bio.protenix as P
    patched = [(m, m._linear) for m in (T, P) if hasattr(m, "_linear")]
    for mod, _ in patched:
        mod._linear = spy   # the switched call sites only; ttnn.linear itself is untouched
    try:
        one_fold, meta, _state = B.build_fold(a.model, ROOT / ".msa_census", a.target, a.a3m)
        t, metrics = one_fold()
    finally:
        for mod, orig in patched:
            mod._linear = orig

    rows = []
    for key, n in counts.most_common():
        mod, fn, line, mt, kt, nt, fired = key
        r = dict(module=mod, func=fn, line=line, mt=mt, kt=kt, nt=nt,
                 tile_macs=mt * kt * nt, fired=fired, calls=n)
        if key in acc:
            r.update(acc[key])
        rows.append(r)

    fired = [r for r in rows if r["fired"]]
    out = dict(model=a.model, target=str(a.target), recycles=a.recycles,
               steps=a.steps, fold_s=round(t, 2), plddt=metrics.get("plddt"),
               n_tokens=metrics.get("n_tokens"),
               grid=list(T.COMPUTE_GRID_MAIN),
               distinct_sites=len(rows), fired_sites=len(fired),
               calls_total=sum(r["calls"] for r in rows),
               calls_fired=sum(r["calls"] for r in fired), rows=rows)
    a.out.write_text(json.dumps(out, indent=1))

    w = "  {:<22} {:>5} {:>6} {:>4} {:>4} {:>9} {:>6} {:>5} {:>10} {:>10} {:>9}"
    print(f"\nfold {t:.1f}s  pLDDT {metrics.get('plddt')}  tokens {metrics.get('n_tokens')}  "
          f"grid {T.COMPUTE_GRID_MAIN}")
    print(f"{len(fired)}/{len(rows)} distinct sites fire, "
          f"{out['calls_fired']}/{out['calls_total']} calls\n")
    print(w.format("func", "line", "calls", "mt", "kt", "tile_macs", "nt", "bw",
                   "max|d|", "rel_max", "rmsd"))
    for r in rows:
        print(w.format(r["func"][:22], r["line"], r["calls"], r["mt"], r["kt"],
                       r["tile_macs"], r["nt"], r.get("in0_block_w", "-"),
                       f"{r['max_abs']:.3e}" if "max_abs" in r else "-",
                       f"{r['rel_max']:.2e}" if "rel_max" in r else "-",
                       f"{r['rmsd']:.3e}" if "rmsd" in r else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

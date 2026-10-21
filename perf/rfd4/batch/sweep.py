"""RFD4 designs/s against the sample width, swept until the card pushes back.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/batch/sweep.py --spec <spec.yaml> --weights <bundle> \
        --out <dir> [--seed 0] [--ladder 1,2,4,8,16,32] [--designs 8] [--reps 2]

Every rung runs the same designs: design k has its own RNG stream (`rng="per_design"`), so rung B
denoises designs 0..N-1 B at a time and design k can be compared bit for bit against the width-1
rung. N is `--designs`, rounded up to a multiple of B (and at least B).

Per rung there is one compile call (the first call at a new width compiles that width's kernels,
it is recorded and is not the warm number), then `--reps` warm calls. A warm line is the whole
span a user waits for: features, all 60 sampler steps at the full 180-pass recipe, the decode and
every CIF written. AICLK is sampled for the life of each call; under 1200 MHz the line is flagged
`clock_artifact`.

The sweep stops when the card pushes back, the same rule the GPU side uses: the chunker had to
halve the width (DRAM refused it), or two rungs in a row gained under 2 % designs/s over the best
so far. Lines go to `<out>/runs.jsonl`; `<out>/summary.json` holds the rung table and the
determinism grade.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402

CLOCK_FLOOR_MHZ = 1200
MIN_GAIN = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder", default="1,2,4,8,16,32")
    ap.add_argument("--designs", type=int, default=8, help="designs per warm call at every rung (rounded up to the width)")
    ap.add_argument("--reps", type=int, default=2, help="warm calls per rung")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import sampler

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "runs.jsonl", "a")
    sha = os.popen("git rev-parse HEAD").read().strip()
    card = os.environ.get("TT_VISIBLE_DEVICES")

    def emit(rec):
        rec.update(sha=sha, card=card, host=os.uname().nodename, t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.perf_counter()
    examples = rfd4.prepare_examples(a.spec, out / "inputs")
    features, atom_array = rfd4.featurize(examples[0].cif, seed=a.seed)
    t_feat = time.perf_counter() - t0
    n_tokens = int(features["padding_mask"].shape[-1])

    with clocksample.during() as clk:
        t0 = time.perf_counter()
        engine = E.DesignEngine(a.weights, recipe="thinking_medium")
        t_build = time.perf_counter() - t0
    emit({"kind": "cold", "what": "engine build (device open + weight upload)", "wall_s": round(t_build, 2),
          "featurize_s": round(t_feat, 2), "aiclk": clk.summary()})

    trunk_s = []
    inner = engine.trunk.run_bucketed

    def timed(*args, **kwargs):
        t = time.perf_counter()
        r = inner(*args, **kwargs)
        trunk_s.append(time.perf_counter() - t)
        return r

    engine.trunk.run_bucketed = timed
    passes = int(sum(sampler.recycle_depths(engine.spec, engine.spec.nsteps)))

    def call(width, n, kind, rep, write=True):
        """One engine call: n designs, `width` at a time, CIFs written; returns (record, output)."""
        engine.max_parallel_samples = width
        trunk_s.clear()
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            output = engine.design(features, num_designs=n, nsteps=engine.spec.nsteps, seed=a.seed, rng="per_design")
            t_design = time.perf_counter() - t0
            t1 = time.perf_counter()
            if write:
                for k in range(n):
                    rfd4.write_design(output, features, atom_array,
                                      out / f"b{width}" / f"{kind}{rep}" / f"{examples[0].example_id}_s{a.seed}_d{k}.cif",
                                      row=k)
            t_write = time.perf_counter() - t1
        summ = clk.summary()
        mhz = _median_mhz(summ)
        wall = t_design + t_write
        rec = {"kind": kind, "size": n_tokens, "batch": width, "width_achieved": engine.chunk_width,
               "designs": n, "rep": rep, "recipe": "thinking_medium", "steps": engine.spec.nsteps,
               "trunk_passes": passes, "network_calls": len(trunk_s), "seed": a.seed, "rng": "per_design",
               "wall_s": round(wall, 3), "design_s": round(t_design, 3), "write_s": round(t_write, 3),
               "trunk_s": round(sum(trunk_s), 3), "host_s": round(t_design - sum(trunk_s), 3),
               "designs_per_s": round(n / wall, 5), "s_per_design": round(wall / n, 3),
               "aiclk": summ, "aiclk_mhz": mhz,
               "clock_artifact": bool(mhz is None or mhz < CLOCK_FLOOR_MHZ)}
        emit(rec)
        return rec, output

    # Width 1 pays the first compile of everything (the AE decode, the glue's first touch).
    call(1, 1, "compile", 0, write=False)

    ladder = [int(x) for x in a.ladder.split(",")]
    rungs, ref, best, flat = [], None, 0.0, 0
    for width in ladder:
        n = max(width, width * math.ceil(a.designs / width))
        try:
            call(width, width, "compile", 0, write=False)
            warm = [call(width, n, "warm", r) for r in range(a.reps)]
        except Exception as exc:  # noqa: BLE001 -- the sweep records why a rung failed, then stops
            emit({"kind": "error", "batch": width, "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
            rungs.append({"batch": width, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            break
        recs = [r for r, _ in warm]
        outs = [o for _, o in warm]
        dps = max(r["designs_per_s"] for r in recs)
        rung = {"batch": width, "width_achieved": min(r["width_achieved"] for r in recs), "designs": n,
                "designs_per_s": dps, "designs_per_s_reps": [r["designs_per_s"] for r in recs],
                "aiclk_mhz": [r["aiclk_mhz"] for r in recs],
                "clock_artifact": any(r["clock_artifact"] for r in recs),
                "trunk_s": [r["trunk_s"] for r in recs], "host_s": [r["host_s"] for r in recs],
                "write_s": [r["write_s"] for r in recs],
                "repeatable": _same(outs[0], outs[-1], n)["bit_exact"]}
        if ref is None:
            ref = outs[0]
        rung["vs_width_1"] = _same(ref, outs[0], min(n, ref.coords_nm.shape[0]))
        torch.save({"coords_nm": outs[0].coords_nm, "residue_type": outs[0].residue_type,
                    "atom_mask": outs[0].atom_mask}, out / f"b{width}" / "designs.pt")
        rungs.append(rung)
        emit({"kind": "rung", **rung})
        if rung["width_achieved"] < width:
            break  # DRAM refused this width; the chunker already ran it narrower
        flat = flat + 1 if dps < best * (1 + MIN_GAIN) else 0
        best = max(best, dps)
        if flat >= 2:
            break

    good = [r for r in rungs if "error" not in r and not r["clock_artifact"]]
    b1 = next((r for r in good if r["batch"] == 1), None)
    top = max(good, key=lambda r: r["designs_per_s"]) if good else None
    summary = {"size": n_tokens, "recipe": "thinking_medium", "trunk_passes": passes, "seed": a.seed,
               "sha": sha, "card": card, "host": os.uname().nodename, "rungs": rungs,
               "best_batch": top and top["width_achieved"],
               "designs_per_s_b1": b1 and b1["designs_per_s"],
               "designs_per_s_best": top and top["designs_per_s"],
               "speedup": (top and b1) and round(top["designs_per_s"] / b1["designs_per_s"], 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rungs"}), flush=True)


def _same(a, b, n):
    """Design k of `a` against design k of `b`, k < n: bit-exactness and the deviation in Angstrom."""
    ca_a, ca_b = a.coords_nm[:n, :, 1].double() * 10, b.coords_nm[:n, :, 1].double() * 10
    m = a.atom_mask[:n].bool() & b.atom_mask[:n].bool()
    d = (a.coords_nm[:n].double() - b.coords_nm[:n].double()).norm(dim=-1) * 10
    ca_rmsd = ((ca_a - ca_b).pow(2).sum(-1).mean(-1)).sqrt()
    return {"designs": n,
            "bit_exact": bool(torch.equal(a.coords_nm[:n], b.coords_nm[:n])
                              and torch.equal(a.residue_type[:n], b.residue_type[:n])),
            "residue_types_equal": bool(torch.equal(a.residue_type[:n], b.residue_type[:n])),
            "max_atom_dev_a": round(float(d[m].max()) if m.any() else 0.0, 4),
            "ca_rmsd_a_per_design": [round(float(x), 4) for x in ca_rmsd],
            "ca_rmsd_a_max": round(float(ca_rmsd.max()), 4)}


def _median_mhz(summary):
    try:
        dev = summary.get(0) or summary.get("0") or next(iter(summary.values()))
        return dev.get("median")
    except Exception:
        return None


if __name__ == "__main__":
    main()

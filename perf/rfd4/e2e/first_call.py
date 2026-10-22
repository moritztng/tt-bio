"""Where does a process's first design part from its warm repeats? (state/rfd4/bets/cells.md)

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/first_call.py --spec <spec.yaml> --weights <bundle> \
        --out <dir> [--designs 3] [--rng upstream]

Runs `--designs` designs at one seed in one process and records, for every trunk call, a float64
digest of its host inputs and of the s it hands back. The first call whose inputs agree across
designs but whose output does not is a device seam; a call whose inputs already differ points at
the host. Then it replays design 0's first trunk call on the warm process: if the replay matches
the warm designs and not design 0, the trunk computes differently on its first call.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()


def digest(t):
    if t is None:
        return None
    x = t.detach().double().flatten()
    w = torch.arange(1, x.numel() + 1, dtype=torch.float64) % 9973
    return [float(x.sum()), float(x.abs().sum()), float((x * w).sum())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--designs", type=int, default=3)
    ap.add_argument("--rng", default="upstream", choices=("upstream", "per_design"))
    a = ap.parse_args()

    from tt_bio import rfd4, tenstorrent as T
    from tt_bio.rfd4 import engine as E

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    designs = rfd4.prepare_examples(a.spec, out / "inputs")
    features, atom_array = rfd4.featurize(designs[0].cif, seed=a.seed)
    engine = E.DesignEngine(a.weights)
    trunk = engine.trunk
    launch, collect = trunk.launch, trunk.collect
    calls, first = [], {}

    def memo():
        return {k: repr(getattr(T, k)) for k in dir(T)
                if k.isupper() and isinstance(getattr(T, k), (dict, set, list)) and len(getattr(T, k)) < 200}

    def rec_launch(s_init, z_init, conds, mask, device, pair=True):
        calls.append({"in": [digest(s_init), digest(z_init), [digest(c) for c in conds], digest(mask)],
                      "pair": pair})
        if "args" not in first:
            first["args"] = (s_init.clone(), z_init.clone(), [c.clone() for c in conds], mask.clone(), pair)
        return launch(s_init, z_init, conds, mask, device, pair=pair)

    def rec_collect(pending):
        s, z = collect(pending)
        calls[-1]["out"] = [digest(s), digest(z)]
        if "s" not in first:
            first["s"] = s.clone()
        return s, z

    trunk.launch, trunk.collect = rec_launch, rec_collect
    per_design, memos, cifs = [], [memo()], []
    for i in range(a.designs):
        calls.clear()
        output = engine.design(features, num_designs=1, nsteps=engine.spec.nsteps, seed=a.seed, rng=a.rng)
        path = rfd4.write_design(output, features, atom_array, out / f"r{i}.cif")
        per_design.append(list(calls))
        memos.append(memo())
        cifs.append(digest(output.coords_nm))
        print(json.dumps({"design": i, "calls": len(calls), "coords": cifs[-1]}), flush=True)

    s0, z0, c0, m0, p0 = first["args"]
    replay, _ = trunk.run_bucketed(s0, z0, c0, m0, engine._device, pair=p0)

    def first_diff(x, y, key):
        for k, (u, v) in enumerate(zip(x, y)):
            if u[key] != v[key]:
                return k
        return None

    report = {"spec": a.spec, "seed": a.seed, "rng": a.rng, "coords": cifs,
              "calls_per_design": [len(c) for c in per_design],
              "first_input_diff_vs_r0": [first_diff(per_design[0], c, "in") for c in per_design[1:]],
              "first_output_diff_vs_r0": [first_diff(per_design[0], c, "out") for c in per_design[1:]],
              "first_output_diff_r1_vs_rest": [first_diff(per_design[1], c, "out") for c in per_design[2:]],
              "replay_call0_vs_r0_call0_maxabs": float((replay - first["s"]).abs().max()),
              "replay_call0_digest": digest(replay),
              "r1_call0_out": per_design[1][0]["out"][0] if len(per_design) > 1 else None,
              "memo_changes": [{k: (m0.get(k), m1.get(k)) for k in set(m0) | set(m1) if m0.get(k) != m1.get(k)}
                               for m0, m1 in zip(memos, memos[1:])],
              "calls": per_design}
    (out / "first_call.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "calls"}, indent=1), flush=True)


if __name__ == "__main__":
    main()

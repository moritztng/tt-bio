"""The throughput table `docs/rfd3-design.md` publishes, re-measured with the kernels default-on.

That table is a *default-configuration* claim, so flipping `RFD3_SPARSE_BIAS` /
`RFD3_FUSED_SCORES` on invalidates it and it has to be regenerated in the same change. This
measures both arms on the same chip in the same process, so each row's change is attributable to
the kernels: the published numbers were measured on a qb1 p150a and this runs on one chip of a
qb2 P300, and a cross-host delta is not a delta.

The timed region is exactly what `tt_bio/rfd3/design.py` times per design: one
`RFD3Sampler.sample` at 200 timesteps (199 transitions, no projection from a short run), with
the same featurize -> token-initializer -> sampler wiring, the same `effective_batch` rule, and
one generator per design in the chunk. Model and features load once per point, and every
(arm, batch) gets its own 4-step warmup so a first-call compile is never inside a timed region.

Arms alternate within a round and the round order flips: two interleaved rounds per point, the
same protocol the published table used. Rep 0 vs rep 1 of one arm is the A/A floor.

Both arms of a round draw from the same seeds, so their outputs must be `torch.equal`. That is
checked at every point and batch, which is a stronger parity statement than the 20-step CIF sha
sweep: byte-identical 199-step trajectories at five design sizes and two batch sizes.

Resumable: one JSON line per timed run, and a run already in the file is skipped.

    /home/ttuser/.coworker/scripts/benchlock.sh rfd3-host-half-defaults-on -- \\
      env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half-defaults-on \\
        PYTHONPATH=$PWD /home/ttuser/tt-bio-dev/env/bin/python3 \\
        scripts/rfd3_port/p44_throughput_table.py --out perf/p44/throughput.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
MPRO = ROOT / "scripts/rfd3_port/parity_artifacts/enzyme_mpro/spec.json"
CKPT = Path("~/.boltz/rfd3/weights").expanduser()

# name, spec dict (or {"__spec__": path}), the atom count the doc's row claims
POINTS = [
    ("40 residues", {"input": str(PDB), "contig": "A1-10,20,A31-40"}, 419),
    ("80 residues", {"input": str(PDB), "contig": "A1-10,60,A31-40"}, 979),
    ("150 residues", {"input": str(PDB), "contig": "A1-10,130,A31-40"}, 1959),
    ("Mpro + nirmatrelvir", {"__spec__": str(MPRO)}, 2702),
    ("250 residues", {"input": str(PDB), "contig": "A1-10,230,A31-40"}, 3359),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "perf/p44/throughput.jsonl")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--points", default="", help="comma-separated point index subset")
    ap.add_argument("--worker", action="store_true", help="internal: run one point")
    ap.add_argument("--point", type=int, default=-1)
    a = ap.parse_args()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    want = [int(i) for i in a.points.split(",")] if a.points else list(range(len(POINTS)))

    if not a.worker:
        # One process per point: the weights load once per point and the device is closed in
        # between, so a large point cannot inherit a small one's fragmented allocator.
        for i in want:
            cmd = [sys.executable, __file__, "--worker", "--point", str(i), "--out", str(a.out),
                   "--timesteps", str(a.timesteps), "--warmup", str(a.warmup),
                   "--rounds", str(a.rounds), "--batches", *[str(b) for b in a.batches]]
            print(f"[point {i}] rc={subprocess.run(cmd, cwd=str(ROOT)).returncode}", flush=True)
        return

    import torch
    from tt_bio import rfd3_bias
    from tt_bio.rfd3.design import (_BATCH_ATOM_PAIR_BUDGET, _BATCH_DESIGN_CEILING,
                                    build_diffusion_module, build_token_initializer)
    from tt_bio.rfd3.featurize import featurize
    from tt_bio.rfd3.input import InputSpecification
    from tt_bio.rfd3.sampler import RFD3Sampler

    name, spec_data, want_atoms = POINTS[a.point]
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["point"], r["batch"], r["arm"], r["round"]))

    if "__spec__" in spec_data:
        path = Path(spec_data["__spec__"])
        spec_data = json.loads(path.read_text())
        inp = Path(spec_data["input"])
        spec_data["input"] = str((inp if inp.is_absolute() else path.parent / inp).resolve())
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()

    dm_w = torch.load(CKPT / "diffusion_module.real_weights.pt", map_location="cpu",
                      weights_only=True)
    ti_w = torch.load(CKPT / "token_initializer.real_weights.pt", map_location="cpu",
                      weights_only=True)
    dev_ti = build_token_initializer(ti_w)
    dev_dm = build_diffusion_module(dm_w)

    f = featurize(spec.input, spec)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    tokens = int(f["restype"].shape[0])
    assert L == want_atoms, f"{name}: {L} atoms, the doc's row claims {want_atoms}"
    print(f"[{name}] I={tokens} atoms={L}", flush=True)

    def arm(on: bool) -> None:
        rfd3_bias.set_enabled(on)
        rfd3_bias.set_fused_enabled(on)

    def sample(batch: int, steps: int, seed0: int):
        sampler = RFD3Sampler(num_timesteps=steps)
        gens = [torch.Generator().manual_seed(seed0 + i) for i in range(batch)]
        with torch.no_grad():
            t0 = time.perf_counter()
            X, _ = sampler.sample(dev_dm, batch, L, coord0, f, init, is_motif, generator=gens)
            return time.perf_counter() - t0, X

    for batch_req in a.batches:
        # The CLI's own ceiling, so the row is what `tt-bio design --batch_size N` really runs.
        batch = min(batch_req, _BATCH_DESIGN_CEILING,
                    max(1, _BATCH_ATOM_PAIR_BUDGET // max(1, L * L)))
        for on in (False, True):
            arm(on)
            sample(batch, a.warmup, 1000 + batch)
        for rnd in range(a.rounds):
            outs = {}
            for tag in (("off", "on") if rnd % 2 == 0 else ("on", "off")):
                arm(tag == "on")
                key = (a.point, batch, tag, rnd)
                if key in done:
                    print(f"[{name}] b={batch} {tag} r={rnd} already measured", flush=True)
                    continue
                before = (rfd3_bias.STATS[0], rfd3_bias.FSTATS[0])
                elapsed, X = sample(batch, a.timesteps, 2000 + 100 * rnd)
                outs[tag] = X
                rec = {"point": a.point, "name": name, "atoms": L, "tokens": tokens,
                       "batch": batch, "batch_requested": batch_req, "arm": tag, "round": rnd,
                       "timesteps": a.timesteps, "elapsed_s": elapsed,
                       "ms_per_step": elapsed / (a.timesteps - 1) * 1000,
                       "ms_per_step_per_design": elapsed / (a.timesteps - 1) / batch * 1000,
                       "designs_per_sec": batch / elapsed, "s_per_design": elapsed / batch,
                       "finite": bool(torch.isfinite(X).all().item()),
                       "sparse_served": rfd3_bias.STATS[0] - before[0],
                       "fused_served": rfd3_bias.FSTATS[0] - before[1],
                       "rejects": {str(k): v for k, v in rfd3_bias.REJECTS.items()}}
                with a.out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[{name}] b={batch} {tag} r={rnd} {elapsed:8.2f}s "
                      f"{rec['designs_per_sec']:.4f} designs/s "
                      f"served={rec['sparse_served']}/{rec['fused_served']}", flush=True)
            if len(outs) == 2:
                eq = bool(torch.equal(outs["off"], outs["on"]))
                mx = (outs["off"] - outs["on"]).abs().max().item()
                print(f"[{name}] b={batch} r={rnd} PARITY equal={eq} maxabs={mx}", flush=True)
                with a.out.open("a") as fh:
                    fh.write(json.dumps({"point": a.point, "name": name, "atoms": L,
                                         "batch": batch, "round": rnd, "arm": "parity",
                                         "equal": eq, "maxabs": mx}) + "\n")


if __name__ == "__main__":
    main()

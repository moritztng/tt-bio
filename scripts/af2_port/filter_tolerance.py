"""What does 0.0846 of i_pTM cost PXDesign's filter?

The device trunk misses the reference by 0.0846 i_pTM, 0.0816 i_pAE and 0.0116 pLDDT
(`parity_artifacts/laczc128_b80/device_trunk_complex.json`). Whether that matters is not a
question about the model, it is a question about the filter: `af2_easy` is a conjunction of
thresholds (pLDDT > 0.8, i_pTM > 0.5, i_pAE < 0.35, bound-unbound RMSD < 3.5) and PXDesign's
success counter is how many designs clear it. A miss only costs something if it moves designs
across those lines.

Nobody had measured that, so this builds a design population that crosses the boundary and
counts the crossings. The population is a sequence ladder on one native interface: chain A is the
`laczc_128` crop, chain B is the 80 residues that follow it in `laczc_256`, both in their native
coordinates, so the pair is a real, well-packed interface. Progressively randomising the binder
sequence while keeping its backbone as the template and the initial guess is exactly the input
AF2-IG sees from PXDesign, a design's own coordinates plus a sequence that is good or bad, and it
sweeps the confidence metrics through the thresholds.

That population turned out not to answer the question: a sequence-contiguous chain split is not
an interface, so all nine rungs are rejected by a wide margin on all three criteria and the
ladder is flat (`parity_artifacts/laczc128_b80/filter_tolerance_scramble.json`). `--mode pose`
is the second population and it varies the one thing that was already known to sit on the
boundary. The port's own parity fixture -- the laczc_128 crop as chain A, its first 80 residues
as chain B translated 60 A clear -- scores i_pTM 0.490, ten thousandths below the accept line,
so walking that translation in from 60 A toward contact sweeps a real pose coordinate through
the threshold with the sequence held native. One variable, no generator.

Each rung is scored by the torch reference arm (bfloat16 trunk, four passes), then re-scored with
the measured device deltas added, and the two verdicts are compared.

    PYTHONPATH=. env/bin/python3 -u scripts/af2_port/filter_tolerance.py --mode pose --out /tmp/tol.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from af2_fixture import read_crop_cif, seq_of, write_complex_pdb  # noqa: E402

TARGET_CIF = REPO / "perf" / "pxdesign" / "targets" / "laczc_256.cif"
FIXTURE_CIF = REPO / "perf" / "pxdesign" / "targets" / "laczc_128.cif"
FIXTURE_BINDER = 80         # == the parity fixture, af2_fixture.build_fixture's default
TARGET_SLICE = (64, 192)    # == laczc_128.cif, verified equal residue for residue
BINDER_SLICE = (192, 256)   # the 64 residues that follow the crop, native pose
DEFAULT_PARAMS = "/home/ttuser/pxd_tool_weights/af2/params_model_1_ptm.npz"

# pxdbench/pxd_configs/eval.py:96-108. af2_easy is the success counter.
AF2_EASY = {"plddt": (">", 0.8), "i_ptm": (">", 0.5), "i_pae": ("<", 0.35)}

# device minus reference, from parity_artifacts/laczc128_b80/device_trunk_complex.json
DEVICE_DELTA = {"plddt": +0.011600, "i_ptm": -0.084555, "i_pae": +0.081586}

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def mutate(sequence: str, fraction: float, seed: int) -> str:
    """Randomise `fraction` of the positions. Seeded, so a rung is reproducible."""
    if fraction <= 0:
        return sequence
    rng = np.random.default_rng(seed)
    out = list(sequence)
    count = int(round(fraction * len(sequence)))
    for i in rng.choice(len(sequence), size=count, replace=False):
        out[i] = AA20[int(rng.integers(20))]
    return "".join(out)


def min_interchain_distance(target_res: list[dict], binder_res: list[dict], shift: float) -> float:
    """Closest approach between the two chains, so a rung records its own geometry."""
    t = np.array([[a[2], a[3], a[4]] for r in target_res for a in r["atoms"]])
    b = np.array([[a[2], a[3], a[4]] for r in binder_res for a in r["atoms"]])
    b = b + np.array([shift, 0.0, 0.0])
    return float(np.linalg.norm(t[:, None, :] - b[None, :, :], axis=-1).min())


def scramble_population(work: Path, levels: list[float], seed: int):
    """Pass 7's population: one native interface, binder sequence randomised in nine steps."""
    residues = read_crop_cif(str(TARGET_CIF))
    target = residues[TARGET_SLICE[0]:TARGET_SLICE[1]]
    binder = residues[BINDER_SLICE[0]:BINDER_SLICE[1]]
    native = seq_of(binder)
    pdb = str(work / "native_complex.pdb")
    write_complex_pdb(pdb, target, binder, shift=0.0)
    for level in levels:
        seq = mutate(native, level, seed + int(level * 1000))
        yield {"level": level,
               "identity": sum(a == b for a, b in zip(seq, native)) / len(native)}, pdb, seq


def pose_population(work: Path, shifts: list[float]):
    """The pose ladder: the parity fixture's binder walked in from 60 A, sequence held native."""
    residues = read_crop_cif(str(FIXTURE_CIF))
    binder = residues[:FIXTURE_BINDER]
    native = seq_of(binder)
    for shift in shifts:
        pdb = str(work / ("pose_%05.1f.pdb" % shift))
        write_complex_pdb(pdb, residues, binder, shift=shift)
        yield {"level": shift,
               "shift_a": shift,
               "min_interchain_a": round(min_interchain_distance(residues, binder, shift), 2)}, \
              pdb, native


def passes(scalars: dict) -> dict:
    return {k: (scalars[k] > bar if op == ">" else scalars[k] < bar)
            for k, (op, bar) in AF2_EASY.items()}


def score(model, pdb: str, binder_seq: str, recycles: int) -> dict:
    from tt_bio.af2_confidence import confidence_scalars
    from tt_bio.af2_data import complex_features, initial_recycle_state
    from tt_bio.af2_reference import run_recycles

    feats_np = complex_features(pdb, binder_seq)
    prev_np = initial_recycle_state(feats_np)

    def to_torch(a):
        if a.dtype == np.bool_:
            return torch.from_numpy(a)
        if a.dtype.kind in "iu":
            return torch.from_numpy(a.astype(np.int64))
        return torch.from_numpy(a.astype(np.float32))

    feats = {k: to_torch(v) for k, v in feats_np.items()}
    prev = {k: to_torch(v) for k, v in prev_np.items()}
    last = None
    for out in run_recycles(model, feats, prev, num_recycles=recycles):
        last = out
    return confidence_scalars(last["plddt_logits"], last["pae_logits"], last["pae_breaks"],
                              feats["seq_mask"], feats["asym_id"],
                              binder_len=len(binder_seq))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--mode", default="scramble", choices=["scramble", "pose"])
    ap.add_argument("--levels", default=None,
                    help="scramble: mutation fractions. pose: binder shifts in angstrom.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--work", default="/tmp/af2ig_tolerance")
    args = ap.parse_args()

    default_levels = {"scramble": "0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0",
                      "pose": "60,50,46,44,42,40,38,36,34,32"}
    levels = [float(x) for x in (args.levels or default_levels[args.mode]).split(",")]

    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    population = (scramble_population(work, levels, args.seed) if args.mode == "scramble"
                  else pose_population(work, levels))

    model = load_af2_model(load_af2_state_dict(args.params), template=True,
                           trunk_dtype=torch.bfloat16)

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["level"] for line in out.read_text().splitlines() if line.strip()}

    for label, pdb, seq in population:
        if label["level"] in done:
            continue
        t0 = time.time()
        ref = score(model, pdb, seq, args.recycles)
        dev = {k: ref[k] + d for k, d in DEVICE_DELTA.items()}
        row = {
            "mode": args.mode,
            **label,
            "seconds": round(time.time() - t0, 1),
            "ref": {k: round(v, 6) for k, v in ref.items()},
            "dev": {k: round(v, 6) for k, v in dev.items()},
            "ref_pass": passes(ref),
            "dev_pass": passes(dev),
        }
        row["flipped"] = sorted(k for k in AF2_EASY if row["ref_pass"][k] != row["dev_pass"][k])
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

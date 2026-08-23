"""What does one AF2-IG design fold cost, and where inside it does the time go?

`trunk_timing.py` times ONE trunk pass. A design is four (`--recycles 3`), plus the embeddings,
the memoised template, the structure module and the two confidence heads -- the exact sequence
`filter_tolerance.py --arm device` runs through `af2_reference.run_recycles`. The port's only
committed fold cost is 6.0 s a design at the 208-token anchor; the cell that carries a verdict is
848 tokens, and that number did not exist before this harness.

Two readings, both from the same run:

* **the fold**, per recycling pass and per design, warm;
* **the split** into template + embeddings (host), the two device stacks, and the host tail
  (structure module and heads). The device stacks each end in a blocking `to_torch`, so the phase
  boundaries are already synchronisation points and the split costs no extra `synchronize_device`
  -- which matters, because an oversynced screen inflates by ~2x
  (`tt-bio-isolated-op-timing-oversync-inflates-cost`).

`--skip <class>` drops one op class from both device stacks (`AF2PairBlock.skip`) for the cost
census: the incumbent is `--skip none`, and a class's share is the incumbent minus its leg. That is
arithmetically wrong on purpose and the inputs are real coordinates in a nonsense interface, so
**nothing here is an accuracy claim.**

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=. \\
        env/bin/python3 scripts/af2_port/fold_timing.py --pdb /tmp/p5_848.pdb --reps 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def feats_from_pdb(pdb: str, target_chain: str = "A", binder_chain: str = "B") -> tuple[dict, str]:
    """Production featurisation of a two-chain PDB, with the binder sequence read off the file.

    Reading the sequence rather than taking it as an argument is what lets any size built by
    `complex_input.py` be timed without also inventing a matching design sequence.
    """
    from tt_bio.af2_data import complex_features, parse_pdb_chain
    from tt_bio import af2_data

    chain = parse_pdb_chain(pdb, binder_chain)
    keep = chain.mask[:, 0] == 1
    restypes = af2_data._rc.restypes
    binder_seq = "".join(restypes[a] if a < len(restypes) else "A"
                         for a in chain.aatype[keep])
    return complex_features(pdb, binder_seq, target_chain, binder_chain), binder_seq


def _fp32_softmax_stats() -> dict:
    from tt_bio import tenstorrent
    return tenstorrent.FP32_SOFTMAX_STATS


def to_torch(a: np.ndarray) -> torch.Tensor:
    if a.dtype == np.bool_:
        return torch.from_numpy(a)
    if a.dtype.kind in "iu":
        return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))


class Split:
    """Wraps the three seams `AF2Model` exposes and records what each pass spent in them."""

    SEAMS = ("template_embedding", "extra_msa_stack", "evoformer_stack")

    def __init__(self, model):
        self.model = model
        self.rows: list[dict] = []
        self._current: dict = {}
        self._originals = {}
        for name in self.SEAMS:
            self._originals[name] = getattr(model, name)
            setattr(model, name, self._wrap(name, self._originals[name]))

    def _wrap(self, name, fn):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._current[name] = self._current.get(name, 0.0) + time.perf_counter() - start
        return timed

    def restore(self) -> None:
        for name, fn in self._originals.items():
            setattr(self.model, name, fn)

    def pass_start(self) -> None:
        self._current = {}

    def pass_end(self, total: float) -> None:
        row = {name: self._current.get(name, 0.0) for name in self.SEAMS}
        row["device_stacks_s"] = row["extra_msa_stack"] + row["evoformer_stack"]
        row["host_rest_s"] = total - row["device_stacks_s"] - row["template_embedding"]
        row["pass_s"] = total
        self.rows.append(row)


def fold(model, feats, prev, recycles: int, split: Split) -> tuple[list[float], dict]:
    """One design: `recycles + 1` passes, `prev` threaded, then the confidence scalars."""
    from tt_bio.af2_confidence import confidence_scalars

    per_pass, last = [], None
    for _ in range(recycles + 1):
        split.pass_start()
        start = time.perf_counter()
        out = model(feats, prev)
        elapsed = time.perf_counter() - start
        split.pass_end(elapsed)
        per_pass.append(elapsed)
        last = out
        prev = {"prev_msa_first_row": out["msa_first_row"],
                "prev_pair": out["pair"],
                "prev_pos": out["structure"]["final_atom_positions"]}
    # The design's own structure output, hashed. A perf arm that claims bit-exactness has to say
    # so against the coordinates this fixture produces, not only against a tap gate on another one.
    coords = last["structure"]["final_atom_positions"].detach().cpu().to(torch.float32).numpy()
    structure_sha16 = hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest()[:16]
    start = time.perf_counter()
    scalars = confidence_scalars(last["plddt_logits"], last["pae_logits"], last["pae_breaks"],
                                 feats["seq_mask"], feats["asym_id"], binder_len=split.binder_len)
    return per_pass, {"confidence_s": time.perf_counter() - start,
                      "structure_sha16": structure_sha16,
                      "scalars": {k: float(v) for k, v in scalars.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--arm", default="device", choices=["device", "torch"])
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3, help="folds; rep 0 is discarded as cold")
    ap.add_argument("--skip", default="none",
                    help="op class from SUBSTITUTION_CLASSES to drop, or 'none'")
    ap.add_argument("--l1-padded-plan", default="inherit", choices=("inherit", "on", "off"),
                    help="derive AF2's fp32-softmax L1 block from the tile-padded token extent. "
                         "`inherit` leaves the shipped per-block default (on) alone; on/off pin "
                         "every AF2 attention, which is what makes the arm switchable inside one "
                         "process. The protenix filter is not reached either way.")
    ap.add_argument("--triatt-fused", default="inherit",
                    choices=["inherit", "none", "trunk", "all"],
                    help='which pair stacks take the fused SDPA triangle attention: "inherit" follows TT_BIO_TRIATT_FUSED_HIFI, "none" pins the materialised fp32 softmax everywhere, "trunk" is extra_msa+evoformer, "all" adds the template pair stack')
    ap.add_argument("--template-host", action="store_true",
                    help="run the template's pair stack in host torch: the A arm of the seam")
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from tt_bio.af2_weights import load_af2_state_dict
    from tap_gate import DEFAULT_PARAMS

    feats_np, binder_seq = feats_from_pdb(args.pdb)
    feats = {k: to_torch(v) for k, v in feats_np.items()}
    tokens = int(feats["seq_mask"].shape[0])

    load_start = time.perf_counter()
    state = load_af2_state_dict(args.params or DEFAULT_PARAMS)
    if args.arm == "torch":
        from tt_bio.af2_reference import load_af2_model
        model = load_af2_model(state, template=True)
    else:
        from tt_bio.af2 import SUBSTITUTION_CLASSES, load_af2_device_model
        model = load_af2_device_model(state, template=True)
        if args.skip != "none":
            if args.skip not in SUBSTITUTION_CLASSES:
                raise SystemExit(f"--skip {args.skip!r} not in {sorted(SUBSTITUTION_CLASSES)}")
            model.set_skip(SUBSTITUTION_CLASSES[args.skip])
        if args.template_host:
            model.set_template_host(True)
        from tt_bio.af2 import TRIATT_FUSED_ARMS
        model.set_triatt_fused(TRIATT_FUSED_ARMS[args.triatt_fused])
        if args.l1_padded_plan != "inherit":
            model.set_l1_padded_plan(args.l1_padded_plan == "on")
    model.eval()
    model_init_s = time.perf_counter() - load_start

    from tt_bio.af2_data import initial_recycle_state
    prev0 = {k: to_torch(v) for k, v in initial_recycle_state(feats_np).items()}

    split = Split(model)
    split.binder_len = len(binder_seq)
    folds = []
    with torch.no_grad():
        for rep in range(args.reps):
            # The template is memoised per design, so each rep needs its own cache, exactly as a
            # real design does.
            model._template_cache = None if args.arm == "device" else None
            start = time.perf_counter()
            per_pass, tail = fold(model, feats, dict(prev0), args.recycles, split)
            folds.append({"rep": rep, "fold_s": time.perf_counter() - start,
                          "pass_s": per_pass, **tail})
            print(json.dumps({k: v for k, v in folds[-1].items() if k != "scalars"}),
                  file=sys.stderr, flush=True)
    split.restore()

    warm_folds = [f["fold_s"] for f in folds[1:]] or [f["fold_s"] for f in folds]
    # Every pass after the very first: the run's cold cost is the first pass of rep 0, and the
    # remaining passes of rep 0 are already warm.
    all_passes = [p for f in folds for p in f["pass_s"]]
    warm_passes = all_passes[1:] or all_passes
    warm_rows = split.rows[1:] or split.rows
    mean = lambda xs: sum(xs) / len(xs)
    report = {
        "mode": "af2ig_fold_timing",
        "label": args.label or Path(args.pdb).stem,
        "arm": args.arm,
        "skip": args.skip,
        "template_host": args.template_host,
        "triatt_fused": args.triatt_fused,
        "l1_padded_plan": args.l1_padded_plan,
        # 0 means the padded and logical extents never disagreed, so the arm was a no-op and any
        # difference between the legs is drift, not the lever.
        "l1_padded_diverged": _fp32_softmax_stats()["l1_padded_diverged"],
        "pdb": args.pdb,
        "tokens": tokens,
        "binder_residues": len(binder_seq),
        "recycles": args.recycles,
        "passes_per_fold": args.recycles + 1,
        "reps": args.reps,
        "model_init_s": model_init_s,
        "cold_first_pass_s": all_passes[0],
        "fold_s_warm_median": statistics.median(warm_folds),
        "fold_s_warm_all": warm_folds,
        "fold_s_cold": folds[0]["fold_s"],
        "pass_s_warm_mean": mean(warm_passes),
        "pass_s_warm_median": statistics.median(warm_passes),
        "pass_s_warm_min": min(warm_passes),
        "pass_s_warm_max": max(warm_passes),
        "pass_s_warm_spread_pct": 100 * (max(warm_passes) - min(warm_passes)) / mean(warm_passes),
        "pass_s_warm_n": len(warm_passes),
        "split_warm_mean_s": {k: mean([r[k] for r in warm_rows])
                              for k in ("template_embedding", "extra_msa_stack",
                                        "evoformer_stack", "device_stacks_s", "host_rest_s")},
        "confidence_s_warm_mean": mean([f["confidence_s"] for f in folds[1:]] or
                                       [f["confidence_s"] for f in folds]),
        # `filter_tolerance.score()` is complex-only: `bound_unbound_RMSD`'s monomer pass is NOT
        # in this number, and the H200 denominator's af2ig stage IS complex + monomer.
        "structure_sha16_all": sorted({f["structure_sha16"] for f in folds}),
        "monomer_pass_included": False,
        "criteria_scored": 3,
        "folds": folds,
        "per_pass_split": split.rows,
    }
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())

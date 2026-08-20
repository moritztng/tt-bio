"""Capture the exact feature dict ColabDesign feeds AlphaFold2, as a committed parity artifact.

Runs in the external CPU-only JAX env (`~/pxd_af2_cpu` on qb2), never inside tt-bio. `jax` and
`colabdesign` are imported inside `main()`, so importing this file does nothing. tt-bio does not
import it at all: the committed `.npz` is what the featurizer is scored against, the same way
`scripts/rfd3_port/capture_ref_f.py` produces an artifact for a card-free gate.

The features are not assembled in `model._inputs`. `prep_inputs` fills the static part, and
`update_seq`, `update_aatype` and `_update_template` run *inside* the jitted model function
(`colabdesign/af/model.py:157-180`), so `msa_feat`, `target_feat`, `aatype`, the atom14/atom37
index maps and the masked template block only exist as tracers there. ColabDesign has a
`pre_callback` hook that fires at that point with the finished dict, so the capture registers one
and ships the values out with `jax.debug.callback`, which sees concrete arrays at run time. That
captures what the model actually consumed rather than a re-derivation of it.

The callback fires once per recycle. Only the first is kept, because that is the pass the
featurizer produces: the later three differ only in `prev`, which is model output.

Output is a compressed `.npz`, not a `.pt`: the capture env has jax and no torch, and a flat
npz keeps the "assert every key" bar honest. Nested dicts (`batch`, `prev`) are flattened with
`/`.

    ~/pxd_af2_cpu/bin/python scripts/af2_port/capture_ref_features.py \
        --cif perf/pxdesign/targets/laczc_128.cif --binder 80 \
        --out scripts/af2_port/parity_artifacts/laczc128_b80/ref_inputs.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from af2_fixture import build_fixture  # noqa: E402

# PXDesign's production AF2 block, verbatim (`pxdbench/pxd_configs/eval.py:53`), and the
# num_recycles hardcoded at `pxdbench/tools/af2/main_af2_complex.py:76,136`.
PRODUCTION = {
    "protocol": "binder",
    "num_recycles": 3,
    "use_multimer": False,
    "use_initial_guess": True,
    "use_initial_atom_pos": False,
}
PREP = {
    "use_binder_template": True,
    "rm_target_seq": True,
    "rm_target_sc": False,
    "rm_template_ic": True,
}


def _flatten(prefix, value, out):
    """Flatten the input pytree to `/`-joined keys, dropping leaves npz cannot hold.

    Runs at trace time, so the leaves are tracers: filter on `dtype`/type only and never call
    `np.asarray` here. `jax.debug.callback` is what turns them into concrete arrays.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), v, out)
        return
    if value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float, bool)):
        out[prefix] = value
        return
    dtype = getattr(value, "dtype", None)
    if dtype is None or dtype.kind not in "biuf":
        return
    out[prefix] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--params", default=os.path.expanduser("~/pxd_tool_weights/af2"))
    ap.add_argument("--work", default="/tmp/af2_capture_work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    args = ap.parse_args()

    import jax
    import numpy as np
    from colabdesign import clear_mem, mk_afdesign_model

    os.makedirs(args.work, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixture = build_fixture(args.cif, os.path.join(args.work, "complex.pdb"), args.binder)
    print(json.dumps(fixture, indent=1), flush=True)

    captured: dict[str, np.ndarray] = {}

    def pre_callback(inputs, **_):
        # Fires inside jit, once per recycle. `jax.debug.callback` hands the leaves over as
        # concrete arrays at run time; the first call is the one the featurizer has to match.
        flat: dict[str, object] = {}
        _flatten("", inputs, flat)

        def receive(values):
            if captured:
                return
            captured.update({k: np.asarray(v) for k, v in values.items()})

        jax.debug.callback(receive, flat)

    clear_mem()
    t0 = time.perf_counter()
    model = mk_afdesign_model(
        data_dir=args.params, pre_callback=pre_callback,
        **{k: v for k, v in PRODUCTION.items() if k != "num_recycles"},
        num_recycles=PRODUCTION["num_recycles"],
    )
    print(f"model constructed in {time.perf_counter() - t0:.2f} s", flush=True)

    if args.stage == "complex":
        model.prep_inputs(pdb_filename=fixture["pdb"], chain="A", binder_chain="B", **PREP)
    else:
        model.prep_inputs(pdb_filename=fixture["pdb"], chain="B")

    t0 = time.perf_counter()
    model.predict(seq=fixture["binder_seq"], models=[0],
                  num_recycles=PRODUCTION["num_recycles"], verbose=False)
    print(f"predict {time.perf_counter() - t0:.2f} s, captured {len(captured)} arrays", flush=True)
    assert captured, "pre_callback never delivered a value"

    log = {k: float(v) for k, v in model.aux["log"].items()
           if isinstance(v, (int, float)) or getattr(v, "ndim", 1) == 0}
    print(json.dumps(log, indent=1, sort_keys=True), flush=True)

    payload = dict(captured)
    payload["_meta/json"] = np.frombuffer(
        json.dumps({
            "fixture": fixture, "production": PRODUCTION, "prep": PREP, "stage": args.stage,
            "jax_version": jax.__version__, "log": log,
            "cif": os.path.basename(args.cif),
        }, sort_keys=True).encode(), dtype=np.uint8)
    # prev_pair is 208x208x128 of zeros on the first recycle, so compression takes the
    # artifact from 22.9 MB to 103 KB and makes it reasonable to commit.
    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    for key in sorted(captured):
        print(f"  {key} {captured[key].shape} {captured[key].dtype}")


if __name__ == "__main__":
    main()

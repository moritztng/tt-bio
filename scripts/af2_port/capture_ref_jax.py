"""Tap AlphaFold2's intermediate activations from the real ColabDesign forward pass.

Runs in the external CPU-only JAX env (`~/pxd_af2_cpu`), never inside tt-bio. `jax`,
`haiku` and `colabdesign` are imported inside `main()`; importing this file does nothing and
`tt_bio` never imports it. The committed `.npz` is what `tt_bio/af2_reference.py` is scored
against, the same contract as `capture_ref_features.py`.

Why taps and not just the end scalars: a torch transcription that is only checked on pLDDT and
i_pTM can be wrong in a way two scalars cannot see. The taps make each stage of the trunk
falsifiable on its own, which is the plan's risk 1.

**How the taps get out.** Both stacks run under `layer_stack.layer_stack`, i.e. a scan, so the
48 Evoformer blocks and the 4 extra-MSA blocks are *traced once*. A trace-time call counter
therefore cannot tell block 0 from block 47. `jax.debug.callback(..., ordered=True)` does: it
fires once per scan iteration at run time, in order, so the host-side counter is the block index.
The same counter separates the four recycles, because ColabDesign's `recycle_mode="last"` loops
the model function in python (`colabdesign/af/design.py:147-205`) rather than in graph.

**What is stored.** A pair activation is 208x208x128 fp32 = 22 MB, and there are dozens of taps,
so storing them whole would be a 100 MB+ artifact. Each tap is stored as full-array statistics
(mean, std, min, max, sum of squares) plus either the whole tensor when it is small or a
self-describing 8192-element subsample -- the element indices are stored next to the values, so
the gate needs no shared RNG and no formula. 8192 samples pin a correlation at the 0.999 bar
several orders of magnitude tighter than the bar itself, and the statistics catch the uniform
shift a correlation cannot see.

    ~/pxd_af2_cpu/bin/python scripts/af2_port/capture_ref_jax.py \
        --cif perf/pxdesign/targets/laczc_128.cif --binder 80 --stage complex \
        --out scripts/af2_port/parity_artifacts/laczc128_b80/ref_taps.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from af2_fixture import build_fixture  # noqa: E402
from capture_ref_features import PREP, PRODUCTION  # noqa: E402

# Store a tensor whole below this many elements, subsample above it. 32768 fp32 is 128 KB.
FULL_MAX = 32768
SUBSAMPLE = 8192

# Which calls of each tap to keep. `None` keeps every call. The Evoformer blocks are indexed
# flat across recycles (48 per recycle), so `< 48` is recycle 0; keeping the first three, the
# middle and the last two is enough to localise a transcription error to a block.
KEEP = {
    "evoformer_iteration": {0, 1, 2, 23, 46, 47},
    "extra_msa_stack": {0, 1, 2, 3},
    "evoformer": {0, 3},          # EmbeddingsAndEvoformer output, first and last recycle
    "structure_module": None,
    "predicted_lddt_head": None,
    "predicted_aligned_error_head": None,
}
# Everything else -- the input/recycling embedder linears, the template stack -- fires once per
# recycle and only recycle 0 is scored, so keep call 0.
KEEP_DEFAULT = {0}

# `common_modules.Linear` and `LayerNorm` are used all over the trunk; tap only the ones that
# are a stage boundary in their own right.
LINEAR_TAPS = {
    "preprocess_1d", "preprocess_msa", "left_single", "right_single", "prev_pos_linear",
    "pair_activiations", "single_activations", "template_single_embedding",
    "template_projection", "extra_msa_activations",
}
LAYERNORM_TAPS = {"prev_msa_first_row_norm", "prev_pair_norm"}

OUT: dict[str, np.ndarray] = {}
COUNTS: dict[str, int] = {}


def _keep(tag: str, n: int) -> bool:
    want = KEEP.get(tag, KEEP_DEFAULT)
    return want is None or n in want


def _numeric(dtype) -> bool:
    """True for anything the store can cast to float32.

    `dtype.kind` is not enough: the trunk runs in bfloat16, and `ml_dtypes.bfloat16` reports
    kind `V`, so a plain kind test silently drops every trunk-internal tap and leaves only the
    float32 boundaries. That is exactly what the first capture did.
    """
    if dtype is None:
        return False
    if getattr(dtype, "kind", "") in "biuf":
        return True
    name = str(dtype)
    return "float" in name or "int" in name or name == "bool"


def _store(key: str, arr) -> None:
    a = np.asarray(arr)
    if not _numeric(a.dtype) or a.size == 0:
        return
    flat = np.asarray(a, dtype=np.float32).reshape(-1)
    OUT[f"{key}/shape"] = np.asarray(a.shape, dtype=np.int64)
    OUT[f"{key}/dtype"] = np.frombuffer(str(a.dtype).encode(), dtype=np.uint8)
    d = flat.astype(np.float64)
    OUT[f"{key}/stats"] = np.asarray(
        [d.mean(), d.std(), d.min(), d.max(), float((d * d).sum())], dtype=np.float64)
    if flat.size <= FULL_MAX:
        OUT[f"{key}/full"] = flat
        return
    idx = np.sort(np.random.default_rng(0).choice(flat.size, SUBSAMPLE, replace=False))
    OUT[f"{key}/idx"] = idx.astype(np.int64)
    OUT[f"{key}/val"] = flat[idx]


def _flat_leaves(prefix: str, value, out: dict) -> None:
    """One level of dict flattening, keeping only numeric leaves small enough to be useful."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flat_leaves(f"{prefix}/{k}" if prefix else str(k), v, out)
        return
    if not _numeric(getattr(value, "dtype", None)):
        return
    if getattr(value, "size", 0) == 0 or value.size > 8_000_000:
        return
    out[prefix] = value


def _emit(jax, tag: str, payload: dict) -> None:
    """Ship one tap's leaves to the host, in scan order, tagged with the call index."""
    if not payload:
        return

    def receive(vals):
        n = COUNTS.get(tag, 0)
        COUNTS[tag] = n + 1
        if not _keep(tag, n):
            return
        for k, v in vals.items():
            _store(f"{tag}#{n}/{k}", v)

    jax.debug.callback(receive, payload, ordered=True)


def _install_taps(jax) -> None:
    from colabdesign.af.alphafold.model import common_modules, folding, modules

    def wrap_named(cls, allowed: set[str], tag_prefix: str):
        original = cls.__call__

        def patched(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            leaf = self.module_name.rsplit("/", 1)[-1]
            if leaf in allowed:
                _emit(jax, f"{tag_prefix}{leaf}", {"out": out})
            return out

        cls.__call__ = patched

    def wrap_module(cls, tag: str):
        original = cls.__call__

        def patched(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            leaves: dict = {}
            _flat_leaves("", out if isinstance(out, dict) else {"out": out}, leaves)
            _emit(jax, tag, leaves)
            return out

        cls.__call__ = patched

    wrap_named(common_modules.Linear, LINEAR_TAPS, "linear/")
    wrap_named(common_modules.LayerNorm, LAYERNORM_TAPS, "norm/")

    # `EvoformerIteration` is instantiated twice, under two haiku names; the tag has to come
    # from the instance, not the class, or the extra-MSA blocks and the Evoformer blocks would
    # share a counter and the block index would be meaningless.
    evo_original = modules.EvoformerIteration.__call__

    def evo_patched(self, *args, **kwargs):
        out = evo_original(self, *args, **kwargs)
        leaves: dict = {}
        _flat_leaves("", out, leaves)
        _emit(jax, self.module_name.rsplit("/", 1)[-1], leaves)
        return out

    modules.EvoformerIteration.__call__ = evo_patched

    wrap_module(modules.TemplatePairStack, "template_pair_stack")
    wrap_module(modules.TemplateEmbedding, "template_embedding")
    wrap_module(modules.EmbeddingsAndEvoformer, "evoformer")
    wrap_module(folding.StructureModule, "structure_module")
    wrap_module(modules.PredictedLDDTHead, "predicted_lddt_head")
    wrap_module(modules.PredictedAlignedErrorHead, "predicted_aligned_error_head")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--params", default=os.path.expanduser("~/pxd_tool_weights/af2"))
    ap.add_argument("--work", default="/tmp/af2_taps_work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    args = ap.parse_args()

    import jax
    from colabdesign import clear_mem, mk_afdesign_model

    _install_taps(jax)

    os.makedirs(args.work, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixture = build_fixture(args.cif, os.path.join(args.work, "complex.pdb"), args.binder)
    print(json.dumps(fixture, indent=1), flush=True)

    production = PRODUCTION[args.stage]
    clear_mem()
    t0 = time.perf_counter()
    model = mk_afdesign_model(data_dir=args.params, **production)
    print(f"model constructed in {time.perf_counter() - t0:.2f} s", flush=True)

    if args.stage == "complex":
        model.prep_inputs(pdb_filename=fixture["pdb"], chain="A", binder_chain="B", **PREP)
    else:
        model.prep_inputs(length=len(fixture["binder_seq"]))

    t0 = time.perf_counter()
    model.predict(seq=fixture["binder_seq"], models=[0],
                  num_recycles=production["num_recycles"], verbose=False)
    print(f"predict {time.perf_counter() - t0:.2f} s", flush=True)

    log = {k: float(v) for k, v in model.aux["log"].items()
           if isinstance(v, (int, float)) or getattr(v, "ndim", 1) == 0}
    print(json.dumps(log, indent=1, sort_keys=True), flush=True)
    print(json.dumps(COUNTS, indent=1, sort_keys=True), flush=True)
    assert OUT, "no tap ever fired"

    OUT["_meta/json"] = np.frombuffer(json.dumps({
        "fixture": fixture, "production": production, "stage": args.stage,
        "prep": PREP if args.stage == "complex" else {},
        "jax_version": jax.__version__, "log": log, "counts": COUNTS,
        "full_max": FULL_MAX, "subsample": SUBSAMPLE,
        "cif": os.path.basename(args.cif),
    }, sort_keys=True).encode(), dtype=np.uint8)
    np.savez_compressed(out_path, **OUT)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB), "
          f"{len(OUT)} arrays", flush=True)
    for key in sorted(k for k in OUT if k.endswith("/shape")):
        base = key[: -len("/shape")]
        how = "full" if f"{base}/full" in OUT else "sub"
        print(f"  {base} {tuple(OUT[key])} {how}")


if __name__ == "__main__":
    main()

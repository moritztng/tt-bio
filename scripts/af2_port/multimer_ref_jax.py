"""Run AF2-multimer_v3's trunk in JAX and dump the fixture and every tap it fired.

Runs in the external CPU-only JAX env (`~/pxd_af2_cpu`), never inside tt-bio, the same contract
as `capture_ref_jax.py`. This one takes the trunk apart rather than driving a design: it builds
`modules_multimer.EmbeddingsAndEvoformer` directly under `hk.transform` with the checkpoint's own
parameters, so the comparison is against AlphaFold's code and not against a second transcription.

The fixture is small on purpose -- two chains of 12 residues, one template, an MSA of depth 1 --
so every tap is stored whole rather than subsampled and the torch arm can be scored element by
element instead of by correlation. The coordinates are a synthetic helix, which keeps every
backbone frame non-degenerate; the template unit vectors are the one feature that would hide a
transcription error behind a NaN if three atoms were collinear.

`--float32` runs the trunk with `global_config.bfloat16` off. That is not what production runs,
and it is the point: this harness answers "is the transform right", and the bf16 question is a
separate measurement against the same taps.

    ~/pxd_af2_cpu/bin/python scripts/af2_port/multimer_ref_jax.py \
        --npz ~/pxd_tool_weights/af2/params_model_1_multimer_v3.npz \
        --out scripts/af2_port/parity_artifacts/multimer_trunk/ref.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tt_bio._vendor.esm.utils import residue_constants as _rc

OUT: dict = {}
COUNTS: dict = {}

#: Where AlphaFold's multimer code lives on this box. BindCraft 2 vendors colabdesign's tree, so
#: the two packages hold the same files; qb1 has colabdesign installed and qb2 has BindCraft 2's
#: copy. The vendored one is preferred because it is the code the design loop differentiates.
AF_PACKAGES = ("bindcraft.af.alphafold.model", "colabdesign.af.alphafold.model")
AF_PACKAGE = [None]


def af_model(*names):
    """Import `names` out of whichever AlphaFold tree is importable, and remember which."""
    import importlib
    for package in AF_PACKAGES:
        try:
            importlib.import_module(package)
        except ModuleNotFoundError:
            continue
        AF_PACKAGE[0] = package
        return tuple(importlib.import_module(f"{package}.{n}") for n in names)
    raise ModuleNotFoundError(
        f"none of {AF_PACKAGES} is importable; put BindCraft 2 on PYTHONPATH "
        "(PYTHONPATH=.:~/bcx_e2e/bc2) or install colabdesign")


#: float64 on the x64 arm, so the reference is not stored through a float32 round trip.
STORE_DTYPE = [np.float32]


def _store(key: str, arr) -> None:
    OUT[key] = np.asarray(arr, dtype=STORE_DTYPE[0])


def _emit(jax, tag: str, payload: dict) -> None:
    def receive(vals):
        n = COUNTS.get(tag, 0)
        COUNTS[tag] = n + 1
        for k, v in vals.items():
            _store(f"{tag}#{n}/{k}", v)

    jax.debug.callback(receive, payload, ordered=True)


def _install_taps(jax) -> None:
    common_modules, modules, modules_multimer = af_model(
        "common_modules", "modules", "modules_multimer")

    def wrap(cls, tag_of):
        original = cls.__call__

        def patched(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            payload = out if isinstance(out, dict) else {"out": out}
            payload = {k: v for k, v in payload.items() if hasattr(v, "dtype")}
            _emit(jax, tag_of(self), payload)
            return out

        cls.__call__ = patched

    def wrap_with_input(cls, tag: str, argument: int):
        """Tap a module's input as well as its output, so a mismatch localises to one side."""
        original = cls.__call__

        def patched(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            act = args[argument] if len(args) > argument else kwargs["act"]
            _emit(jax, tag, {"in": act, "out": out})
            return out

        cls.__call__ = patched

    # The nine template feature embeddings, each tapped under its own haiku name, so a
    # mismatch in the summed input localises to one feature rather than to the sum.
    linear_original = common_modules.Linear.__call__

    def linear_patched(self, *args, **kwargs):
        out = linear_original(self, *args, **kwargs)
        leaf = self.module_name.rsplit("/", 1)[-1]
        if leaf.startswith("template_pair_embedding_"):
            _emit(jax, f"pair_embedding/{leaf}", {"out": out})
        return out

    common_modules.Linear.__call__ = linear_patched

    wrap(modules.EvoformerIteration, lambda m: m.module_name.rsplit("/", 1)[-1])
    wrap_with_input(modules_multimer.TemplateEmbeddingIteration,
                    "template_embedding_iteration", 0)
    wrap(modules_multimer.SingleTemplateEmbedding, lambda m: "single_template_embedding")
    wrap(modules_multimer.TemplateEmbedding, lambda m: "template_embedding")
    wrap(modules_multimer.EmbeddingsAndEvoformer, lambda m: "trunk")


class _Float64Numpy:
    """`jnp` with `float32` reading as `float64`, for the x64 reference arm.

    AlphaFold picks its activation dtype as `bfloat16 if gc.bfloat16 else jnp.float32` and its
    LayerNorm upcasts with `astype(jnp.float32)`, so under `jax_enable_x64` the trunk would
    still narrow to float32 at every one of those sites and the "float64 reference" would be a
    float32 one wearing a wider container. Rebinding the name in the three modules that make
    those choices is the smallest change that makes the arm honest; it touches the external JAX
    environment only, never tt_bio.
    """

    def __init__(self, jnp):
        self._jnp = jnp

    def __getattr__(self, name):
        if name == "float32":
            return self._jnp.float64
        return getattr(self._jnp, name)


def build_fixture(num_res: int, chain_lengths, num_templates: int, seed: int,
                  translate: float = 0.0, chi1_only: bool = False) -> dict:
    """A two-chain fixture with a helical backbone. Every array is what the trunk reads."""
    rng = np.random.default_rng(seed)
    assert sum(chain_lengths) == num_res

    asym, entity, sym, residue_index = [], [], [], []
    for chain, length in enumerate(chain_lengths):
        asym += [chain + 1] * length
        # Both chains are the same entity here only if their lengths match; keep them distinct
        # so `entity_id_same` is not trivially all ones and the relative encoding's extra bins
        # are all exercised.
        entity += [chain + 1] * length
        sym += [1] * length
        residue_index += list(range(length))

    # An alpha helix: 1.5 A rise, 100 degrees per residue, N/CA/C offset around the axis.
    theta = np.arange(num_res) * (100.0 * np.pi / 180.0)
    ca = np.stack([2.3 * np.cos(theta), 2.3 * np.sin(theta), 1.5 * np.arange(num_res)], -1)
    n = ca + np.stack([0.9 * np.cos(theta + 0.6), 0.9 * np.sin(theta + 0.6),
                       -0.6 * np.ones(num_res)], -1)
    c = ca + np.stack([1.1 * np.cos(theta - 0.7), 1.1 * np.sin(theta - 0.7),
                       0.7 * np.ones(num_res)], -1)
    cb = ca + np.stack([1.5 * np.cos(theta + 2.1), 1.5 * np.sin(theta + 2.1),
                        -0.5 * np.ones(num_res)], -1)
    o = c + rng.normal(0, 0.05, (num_res, 3)) + np.array([0.0, 0.0, 1.2])

    positions = np.zeros((num_res, 37, 3), dtype=np.float32)
    mask = np.zeros((num_res, 37), dtype=np.float32)
    for index, coords in ((0, n), (1, ca), (2, c), (3, o), (4, cb)):
        positions[:, index] = coords
        mask[:, index] = 1.0
    # Every remaining atom37 slot, so chi 1 is defined whatever residue is drawn: the
    # side-chain gamma atom sits at a different index for serine, threonine and cysteine than
    # for arginine, and a template row with no chi 1 is masked out of the MSA.
    for index in range(5, 37):
        positions[:, index] = ca + rng.normal(0, 1.4, (num_res, 3))
        mask[:, index] = 1.0

    # Alanine and glycine have no chi 1, so a template row of either puts a zero in multimer_v3's
    # template MSA mask. `chi1_only` draws neither, which is how the ordering delta was measured
    # before the mask was wired up; the default draws all twenty, so the MSA mask this fixture
    # hands the trunk has real zeros in it and the masked path is the one under test.
    restypes = np.arange(20)
    if chi1_only:
        restypes = np.array([r for r in restypes if r not in (_rc.restype_order["A"],
                                                              _rc.restype_order["G"])])
    aatype = restypes[rng.integers(0, len(restypes), num_res)].astype(np.int32)
    template_positions = np.repeat(positions[None], num_templates, 0)
    template_positions += rng.normal(0, 0.15, template_positions.shape).astype(np.float32)

    # Every feature AF2 reads off coordinates is relative, so a rigid translation must not move
    # the trunk's output. It does move the float32 cancellation in the template unit vectors,
    # which is the whole point of the knob: the residual that survives it scales with the
    # offset, and a residual that scales with an offset the model is invariant to is the
    # reference's rounding, not a difference between the two implementations.
    if translate:
        positions = positions + np.float32(translate)
        template_positions = template_positions + np.float32(translate)

    return {
        "aatype": aatype,
        "residue_index": np.asarray(residue_index, dtype=np.int32),
        "asym_id": np.asarray(asym, dtype=np.int32),
        "entity_id": np.asarray(entity, dtype=np.int32),
        "sym_id": np.asarray(sym, dtype=np.int32),
        "seq_mask": np.ones(num_res, dtype=np.float32),
        "msa_mask": np.ones((1, num_res), dtype=np.float32),
        "msa_feat": rng.normal(0, 1, (1, num_res, 49)).astype(np.float32),
        "target_feat": np.eye(20, dtype=np.float32)[aatype],
        "extra_msa": rng.integers(0, 21, (1, num_res)).astype(np.int32),
        "extra_deletion_value": np.zeros((1, num_res), dtype=np.float32),
        # BindCraft 2 hands AF2 an all-zero extra-MSA mask (`bindcraft/af2.py:134`), which is the
        # precondition `tt_bio.af2.AF2DeviceModel` already relies on for the monomer.
        "extra_msa_mask": np.zeros((1, num_res), dtype=np.float32),
        "prev_pos": positions.astype(np.float32),
        "prev_pair": rng.normal(0, 0.5, (num_res, num_res, 128)).astype(np.float32),
        "prev_msa_first_row": rng.normal(0, 0.5, (num_res, 256)).astype(np.float32),
        "template_aatype": np.repeat(aatype[None], num_templates, 0),
        "template_all_atom_positions": template_positions.astype(np.float32),
        "template_all_atom_mask": np.repeat(mask[None], num_templates, 0),
        "template_mask": np.ones(num_templates, dtype=np.float32),
        "mask_template_interchain": np.asarray(1.0, dtype=np.float32),
        "use_dropout": np.asarray(0.0, dtype=np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-res", type=int, default=24)
    ap.add_argument("--chains", default="12,12")
    ap.add_argument("--templates", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--extra-blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--translate", type=float, default=0.0,
                    help="rigidly translate every coordinate by this many Angstrom")
    ap.add_argument("--chi1-only", action="store_true",
                    help="draw no alanine and no glycine, so every template MSA row is "
                         "unmasked; the fixture the ordering delta was first measured on")
    ap.add_argument("--float32", action="store_true",
                    help="run with global_config.bfloat16 off (the transform question)")
    ap.add_argument("--float64", action="store_true",
                    help="float64 throughout: implies --float32 and enables jax x64. This is "
                         "the arm the port is scored against, because float32 is itself an "
                         "approximation and one of its roundings is amplified to O(1) by the "
                         "1e-6 norm clip on the template unit vectors.")
    args = ap.parse_args()

    if args.float64:
        args.float32 = True
        import jax as _jax
        _jax.config.update("jax_enable_x64", True)

    import haiku as hk
    import jax
    import jax.numpy as jnp
    af_config, modules_multimer, prng, utils = af_model(
        "config", "modules_multimer", "prng", "utils")

    if args.float64:
        STORE_DTYPE[0] = np.float64
    _install_taps(jax)

    cfg = af_config.CONFIG_MULTIMER.model.embeddings_and_evoformer
    gc = af_config.CONFIG_MULTIMER.model.global_config
    cfg.evoformer_num_block = args.blocks
    cfg.extra_msa_stack_num_block = args.extra_blocks
    gc.bfloat16 = not args.float32

    with np.load(args.npz, allow_pickle=False) as npz:
        source = {k: npz[k] for k in npz.files}
    prefix = "alphafold/alphafold_iteration/"
    params = {}
    for path, array in source.items():
        scope, name = path.split("//")
        if not scope.startswith(prefix + "evoformer"):
            continue
        scope = scope[len(prefix):]
        params.setdefault(scope, {})[name] = jnp.asarray(
            array, dtype=jnp.float64 if args.float64 else jnp.float32)

    batch = build_fixture(args.num_res, [int(x) for x in args.chains.split(",")],
                          args.templates, args.seed, args.translate, args.chi1_only)

    def forward(batch):
        return modules_multimer.EmbeddingsAndEvoformer(cfg, gc, name="evoformer")(
            batch, safe_key=prng.SafeKey(jax.random.PRNGKey(0)))

    model = hk.transform(forward)
    if args.float64:
        common_modules, modules, modules_multimer = af_model(
            "common_modules", "modules", "modules_multimer")
        wide = _Float64Numpy(jnp)
        for module in (common_modules, modules, modules_multimer):
            module.jnp = wide

    float_dtype = jnp.float64 if args.float64 else jnp.float32
    inputs = {k: jnp.asarray(v, dtype=float_dtype if np.asarray(v).dtype.kind == "f" else None)
              for k, v in batch.items()}
    out = jax.jit(model.apply)(params, jax.random.PRNGKey(0), inputs)
    out = jax.tree_util.tree_map(np.asarray, out)

    for key, value in batch.items():
        OUT[f"batch/{key}"] = np.asarray(value, dtype=np.float32)
    # The fixture is built in float32 and is exact in both arms; only the taps need the width.
    for key, value in out.items():
        _store(f"out/{key}", value)
    OUT["_meta/json"] = np.frombuffer(json.dumps({
        "npz": Path(args.npz).name, "num_res": args.num_res, "chains": args.chains,
        "templates": args.templates, "blocks": args.blocks, "translate": args.translate,
        "extra_blocks": args.extra_blocks, "seed": args.seed,
        "chi1_only": bool(args.chi1_only), "af_package": AF_PACKAGE[0],
        "bfloat16": bool(gc.bfloat16), "float64": bool(args.float64),
        "jax_version": jax.__version__,
        "counts": COUNTS,
    }, sort_keys=True).encode(), dtype=np.uint8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **OUT)
    print(json.dumps(COUNTS, sort_keys=True))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB), {len(OUT)} arrays")


if __name__ == "__main__":
    main()

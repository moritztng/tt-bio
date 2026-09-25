"""Price BindCraft 2's host tail: everything jax.value_and_grad sees that is not the Evoformer trunk.

The differentiated callee is bindcraft/af2.py:352 sequence_design_loss. Its trunk part is
EmbeddingsAndEvoformer (268 of the checkpoint's 330 arrays). Everything after it -- the structure
module, the five heads, the confidence metrics, the template alignment and the active losses -- is
what tt_bio/af2.py does not have. This script runs exactly that region, forward and backward,
with the checkpoint's own parameters, and times it.

No transcription: the modules, the metrics function, the alignment and the losses are imported
from the BindCraft 2 tree at pin 7a2dfdb8a285232a6f881899fe135c6dc48679f1.
"""
import argparse
import functools
import gc
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/tmp/bc2_probe")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from bindcraft import af2 as bc_af2
from bindcraft import loss as bc_loss
from bindcraft.af.alphafold.model import config as af_config
from bindcraft.af.alphafold.model import folding_multimer, modules, prng
from bindcraft.prediction import residue_chain_ids, split_residue_arrays_by_chain
from bindcraft.protein import Protein, ResidueFlags, StructurePrediction

CKPT = os.path.expanduser("~/bcx_tail/params_model_1_multimer_v3.npz")
TAIL_SCOPES = ("structure_module", "distogram_head", "masked_msa_head", "predicted_lddt_head",
               "experimentally_resolved_head", "predicted_aligned_error_head")


def load_params(path=CKPT):
    """The checkpoint, split into trunk and tail by haiku scope."""
    raw = np.load(path)
    trunk, tail = {}, {}
    for key in raw.files:
        scope, name = key.rsplit("//", 1)
        leaf = scope.split("/")[2] if scope.startswith("alphafold/alphafold_iteration/") else None
        target = tail if leaf in TAIL_SCOPES else trunk
        target.setdefault(scope, {})[name] = jnp.asarray(raw[key])
    return trunk, tail


def model_config():
    """af2.py:241-256 -- exactly the global config BindCraft 2 runs the trunk and tail under."""
    cfg = af_config.model_config("model_1_multimer_v3")
    cfg.model.global_config.use_dgram = False
    cfg.model.global_config.use_remat = True
    cfg.model.global_config.bfloat16 = True
    cfg.model.global_config.subbatch_size = None
    cfg.model.global_config.attention_backend = "stock"
    cfg.model.global_config.use_cueq = False
    cfg.model.num_recycle = 0
    return cfg


def tail_forward(cfg):
    """The second half of AlphaFoldIteration.__call__ (modules_multimer.py:112-150).

    Takes the trunk's representations as an argument instead of computing them, so the tail is
    differentiable with respect to the trunk outputs -- which is the cotangent the device trunk
    would have to receive.
    """
    c, gc_ = cfg.model, cfg.model.global_config

    def fn(representations, batch, stop_after=None):
        class Iteration(hk.Module):
            def __init__(self):
                super().__init__(name="alphafold_iteration")

            def __call__(self, representations, batch, stop_after=None):
                representations = dict(representations)
                heads = {}
                factories = {
                    "masked_msa": modules.MaskedMsaHead,
                    "distogram": modules.DistogramHead,
                    "structure_module": folding_multimer.StructureModule,
                    "predicted_aligned_error": modules.PredictedAlignedErrorHead,
                    "predicted_lddt": modules.PredictedLDDTHead,
                    "experimentally_resolved": modules.ExperimentallyResolvedHead,
                }
                for name, head_cfg in sorted(c.heads.items()):
                    if not head_cfg.weight:
                        continue
                    heads[name] = (head_cfg, factories[name](head_cfg, gc_))

                _, fold_module = heads["structure_module"]
                structure_output = fold_module(representations, batch)
                if stop_after == "structure_module":
                    return {"structure_module": structure_output}

                ret = {}
                for name, (_, module) in heads.items():
                    if name == "structure_module":
                        ret[name] = structure_output
                        representations["structure_module"] = structure_output.pop("act")
                    elif name in {"predicted_lddt", "predicted_aligned_error",
                                  "experimentally_resolved"}:
                        continue
                    else:
                        ret[name] = module(representations, batch)

                for name in ("predicted_lddt", "experimentally_resolved",
                             "predicted_aligned_error"):
                    if c.heads.get(f"{name}.weight", 0.0) or (
                            name == "experimentally_resolved"
                            and c.heads.experimentally_resolved.weight):
                        head_cfg, module = heads[name]
                        ret[name] = module(representations, batch)
                if "predicted_aligned_error" in ret:
                    ret["predicted_aligned_error"]["asym_id"] = batch["asym_id"]
                return ret

        class AlphaFold(hk.Module):
            def __init__(self):
                super().__init__(name="alphafold")

            def __call__(self, representations, batch, stop_after=None):
                return Iteration()(representations, batch, stop_after)

        return AlphaFold()(representations, batch, stop_after)

    return hk.transform(fn)


# --------------------------------------------------------------------------------------------
# A BindCraft 2 design state at the n=256 bucket: PD-L1 (115) + a 130-residue binder, padded to
# 256 by af2.py's DEFAULT_LENGTH_BUCKET of 32.
# --------------------------------------------------------------------------------------------
def build_state(n_target=115, n_binder=141, n_binder_real=130, seed=0):
    """A design state as sequence_gradients builds it.

    af2.py:385 pads only the DESIGN chain to the bucket (`pad_design_chains`), and the target is
    left alone at the default `target_pad_length=0`. So the token count is
    (binder rounded up to 32) + target, and the padded residues sit INSIDE the binder chain
    carrying ResidueFlags.PADDING -- not appended after the target.
    """
    n = n_binder + n_target
    real = n_binder_real + n_target
    rng = np.random.default_rng(seed)

    chain_lengths = (n_binder, n_target)  # sorted chain names: 'binder', 'target'
    chain_names = ("binder", "target")

    aatype = rng.integers(0, 20, n).astype(np.int32)
    aatype[n_binder_real:n_binder] = 0
    seq_mask = np.ones(n, dtype=np.float32)
    seq_mask[n_binder_real:n_binder] = 0.0

    asym_id = np.ones(n, dtype=np.int32)
    asym_id[:n_binder] = 0

    residue_index = np.arange(1, n + 1, dtype=np.int32)

    flags = np.empty(n, dtype=np.uint8)
    flags[:n_binder_real] = int(ResidueFlags.DESIGN)
    flags[n_binder_real:n_binder] = int(ResidueFlags.PADDING)
    flags[n_binder:] = int(ResidueFlags.TEMPLATE)

    # A plausible target backbone: a compact random walk, so kabsch and the distance losses see
    # real geometry rather than coincident atoms.
    ca = np.cumsum(rng.normal(0, 1.9, (n, 3)), axis=0).astype(np.float32)
    atoms = np.zeros((n, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((n, 37), dtype=np.float32)
    for idx, offset in ((0, [-1.2, 0.3, 0.0]), (1, [0.0, 0.0, 0.0]),
                        (2, [1.3, 0.4, 0.2]), (3, [1.6, 1.5, 0.5]), (4, [0.7, -1.2, 0.9])):
        atoms[:, idx] = ca + np.asarray(offset, dtype=np.float32)
        atom_mask[:, idx] = 1.0
    # The binder has no template coordinates; the target does. Padding residues have none either.
    atoms[:n_binder] = 0.0
    atom_mask[:n_binder] = 0.0

    sequence = np.zeros((n, 20), dtype=np.float32)
    sequence[np.arange(n), np.minimum(aatype, 19)] = 1.0

    from bindcraft.af.alphafold.common import residue_constants as rc
    mask = seq_mask[:, None]
    batch = {
        "aatype": jnp.asarray(aatype),
        "seq_mask": jnp.asarray(seq_mask),
        "residue_index": jnp.asarray(residue_index),
        "asym_id": jnp.asarray(asym_id),
        "entity_id": jnp.asarray(asym_id),
        "sym_id": jnp.asarray(asym_id),
        "all_atom_positions": jnp.asarray(atoms, dtype=jnp.float32),
        "all_atom_mask": jnp.asarray(atom_mask, dtype=jnp.float32),
        "atom14_atom_exists": jnp.asarray(np.where(mask, rc.restype_atom14_mask[aatype], 0)),
        "atom37_atom_exists": jnp.asarray(np.where(mask, rc.restype_atom37_mask[aatype], 0)),
        "residx_atom14_to_atom37": jnp.asarray(
            np.where(mask, rc.restype_atom14_to_atom37[aatype], 0)),
        "residx_atom37_to_atom14": jnp.asarray(
            np.where(mask, rc.restype_atom37_to_atom14[aatype], 0)),
        "use_dropout": jnp.asarray(False),
    }
    meta = dict(n=n, real=real, chain_names=chain_names, chain_lengths=chain_lengths,
                n_target=n_target, n_binder=n_binder, n_binder_real=n_binder_real,
                # Protein-side dtypes, as bindcraft.protein.Protein stores them: atoms float16,
                # atom_mask bool. af2.py casts to float32 only on the way into the batch.
                sequence=jnp.asarray(sequence, dtype=jnp.float16),
                atoms=jnp.asarray(atoms, dtype=jnp.float16),
                atom_mask=jnp.asarray(atom_mask.astype(bool)), flags=jnp.asarray(flags),
                residue_index=jnp.asarray(residue_index))
    return batch, meta


def random_representations(n, cfg, seed=1):
    """Trunk outputs at the shapes EmbeddingsAndEvoformer produces them.

    Timing is value independent here -- every shape is static under jit and no branch in the tail
    reads a representation value -- so random values price the tail exactly. The `cut` subcommand
    is what checks the structural question about these four outputs.
    """
    c = cfg.model.embeddings_and_evoformer
    rng = np.random.default_rng(seed)
    msa_rows = 1  # BindCraft 2 feeds a single MSA row (af2.py:132 msa_feat is (1, length, 49))
    f = lambda *s: jnp.asarray(rng.normal(0, 1.0, s).astype(np.float32))
    return {
        "single": f(n, c.seq_channel),
        "pair": f(n, n, c.pair_channel),
        "msa": f(msa_rows, n, c.msa_channel),
        "msa_first_row": f(n, c.msa_channel),
    }


BC2_SETTINGS = "/tmp/bc2_probe/settings/core/default.json"


def build_loss_set(path=BC2_SETTINGS):
    """BindCraft 2's own active loss set, from its own settings file.

    build_losses (loss.py:101) keys off every nonzero `weights_<name>`, not off the `losses`
    dict -- the `losses` dict only supplies params. So `iptm_loss` (0.05) and `interface_pae`
    (0.1) are active with their default params even though the `losses` dict does not mention
    them, and the active set is eight, not the six that dict lists.
    """
    with open(path) as fh:
        settings = json.load(fh)
    return bc_loss.build_losses(settings, seed=0), settings


def tail_program(cfg, losses, batch, meta, params_tail, key):
    """representations -> total loss. Exactly the region af2.py runs after the trunk."""
    transformed = tail_forward(cfg)
    chain_names, chain_lengths = meta["chain_names"], meta["chain_lengths"]
    interface_asym_id = bc_af2.interface_asym_ids(chain_names, chain_lengths)

    def program(representations):
        outputs = transformed.apply(params_tail, key, representations, batch)
        positions = outputs["structure_module"]["final_atom_positions"].astype(jnp.float16)
        mask = outputs["structure_module"]["final_atom_mask"].astype(bool)
        metrics = bc_af2.alphafold_prediction_metrics(outputs, batch["seq_mask"],
                                                      interface_asym_id)
        positions = bc_af2.align_prediction_to_target_template(
            positions, mask, meta["atoms"], meta["atom_mask"], meta["flags"])
        input_arrays = split_residue_arrays_by_chain(
            chain_names, chain_lengths, sequence=meta["sequence"], atoms=meta["atoms"],
            atom_mask=meta["atom_mask"], flags=meta["flags"],
            residue_index=meta["residue_index"])
        predicted_arrays = split_residue_arrays_by_chain(
            chain_names, chain_lengths, atoms=positions, atom_mask=mask)
        input_complex = {name: Protein(**input_arrays[name]) for name in chain_names}
        predicted = {name: input_complex[name].replace(**predicted_arrays[name])
                     for name in chain_names}
        states = {"complex": input_complex}
        predictions = {"complex": StructurePrediction(protein_complex=predicted,
                                                      metrics=dict(metrics))}
        total = jnp.asarray(0.0)
        for name, entry in losses.items():
            total = total + entry.weight * entry.function(states, predictions)
        return total

    return program


def timeit(fn, args, repeats, label):
    out = jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return dict(label=label, n=len(samples), median=statistics.median(samples),
                min=min(samples), max=max(samples), samples=[round(s, 5) for s in samples])


def host_stamp():
    load1, load5, load15 = os.getloadavg()
    return dict(host=os.uname().nodename, cores=os.cpu_count(),
                affinity=sorted(os.sched_getaffinity(0)),
                load1=round(load1, 2), load5=round(load5, 2), load15=round(load15, 2),
                threads=os.environ.get("XLA_FLAGS", ""), jax=jax.__version__,
                devices=str(jax.devices()))


def cmd_bind(args):
    """Do the checkpoint's tail arrays bind to this tail-only transform, exactly?"""
    cfg = model_config()
    _, tail = load_params()
    batch, meta = build_state()
    reps = random_representations(meta["n"], cfg)
    transformed = tail_forward(cfg)
    init = transformed.init(jax.random.PRNGKey(0), reps, batch)
    want = {f"{s}//{n}" for s, d in init.items() for n in d}
    have = {f"{s}//{n}" for s, d in tail.items() for n in d}
    print(f"tail transform wants {len(want)} arrays, checkpoint tail holds {len(have)}")
    print(f"missing from checkpoint: {sorted(want - have)}")
    print(f"unused in checkpoint:    {sorted(have - want)}")
    shape_mismatch = [k for k in sorted(want & have)
                      if init[k.split('//')[0]][k.split('//')[1]].shape
                      != tail[k.split('//')[0]][k.split('//')[1]].shape]
    print(f"shape mismatches: {shape_mismatch}")
    ok = not (want - have) and not (have - want) and not shape_mismatch
    print("BIND:", "exact" if ok else "NOT exact")
    return 0 if ok else 1


def _float_sum(tree):
    """Force every float leaf, so XLA cannot DCE a stage whose output we are pricing."""
    total = jnp.asarray(0.0)
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "dtype") and leaf.dtype.kind == "f":
            total = total + leaf.astype(jnp.float32).sum()
    return total


def cmd_time(args):
    cfg = model_config()
    _, tail = load_params()
    batch, meta = build_state(n_target=args.n_target, n_binder=args.n_binder,
                              n_binder_real=args.n_binder_real)
    reps = random_representations(meta["n"], cfg)
    losses, settings = build_loss_set()
    key = jax.random.PRNGKey(0)
    chain_names, chain_lengths = meta["chain_names"], meta["chain_lengths"]
    interface_asym_id = bc_af2.interface_asym_ids(chain_names, chain_lengths)
    transformed = tail_forward(cfg)

    def stage_structure_module(representations):
        return transformed.apply(tail, key, representations, batch, stop_after="structure_module")

    def stage_heads(representations):
        return transformed.apply(tail, key, representations, batch)

    def stage_metrics(representations):
        outputs = stage_heads(representations)
        return bc_af2.alphafold_prediction_metrics(outputs, batch["seq_mask"], interface_asym_id)

    def stage_align(representations):
        outputs = stage_heads(representations)
        metrics = bc_af2.alphafold_prediction_metrics(outputs, batch["seq_mask"],
                                                      interface_asym_id)
        positions = outputs["structure_module"]["final_atom_positions"].astype(jnp.float16)
        mask = outputs["structure_module"]["final_atom_mask"].astype(bool)
        aligned = bc_af2.align_prediction_to_target_template(
            positions, mask, meta["atoms"], meta["atom_mask"], meta["flags"])
        return metrics, aligned

    full = tail_program(cfg, losses, batch, meta, tail, key)

    stages = [
        ("1_structure_module", lambda r: _float_sum(stage_structure_module(r))),
        ("2_plus_heads", lambda r: _float_sum(stage_heads(r))),
        ("3_plus_metrics", lambda r: _float_sum(stage_metrics(r))),
        ("4_plus_align", lambda r: _float_sum(stage_align(r))),
        ("5_full_tail_loss", full),
    ]

    results, compile_s = [], {}
    for label, fn in stages:
        f = jax.jit(fn)
        g = jax.jit(jax.value_and_grad(fn))
        t0 = time.perf_counter(); jax.block_until_ready(f(reps))
        compile_s[f"{label}_forward"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter(); jax.block_until_ready(g(reps))
        compile_s[f"{label}_value_and_grad"] = round(time.perf_counter() - t0, 3)
        results.append(timeit(f, (reps,), args.repeats, f"{label}_forward"))
        results.append(timeit(g, (reps,), args.repeats, f"{label}_value_and_grad"))

    value, grads = jax.block_until_ready(jax.jit(jax.value_and_grad(full))(reps))
    out = dict(stamp=host_stamp(), n=meta["n"], real=meta["real"],
               n_target=meta["n_target"], n_binder=meta["n_binder"],
               n_binder_real=meta["n_binder_real"],
               losses={name: float(entry.weight) for name, entry in sorted(losses.items())},
               loss_value=float(value),
               compile_s=compile_s,
               cotangent_shapes={k: list(v.shape) for k, v in grads.items()},
               cotangent_bytes_bf16={k: int(np.prod(v.shape)) * 2 for k, v in grads.items()},
               timings=results)
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0


def cmd_cut(args):
    """Are the two trunk outputs the tail consumes a graph cut? PROTOCOL A41/D242.

    Injecting cotangents at several outputs at once is only valid if none of them is an
    ANCESTOR of another. A shared ancestor is fine; an ancestor-descendant pair is not.

    The Evoformer block, read off modules.py:1296-1391:

        opm_first=True  (MULTIMER, what BindCraft 2 runs)
            pair_a = pair_0 + OPM(msa_0)          # step 1
            msa_1  = attn/transition(msa_0, pair_a)   # steps 2-4, reads pair_a
            pair_1 = triangles/transition(pair_a)     # step 5, reads pair_a -- NOT msa_1

        opm_first=False (MONOMER, model_1_ptm, what tt_bio/af2.py ships)
            msa_1  = attn/transition(msa_0, pair_0)
            pair_a = pair_0 + OPM(msa_1)          # reads the UPDATED msa
            pair_1 = triangles/transition(pair_a)

    So the ordering flag decides the property. This measures it rather than reading it: a
    tangent is injected at the msa_1 node and we see whether pair_1 moves.
    """
    n, cm, cp, cs = args.n, 64, 32, 48
    rng = np.random.default_rng(0)
    arr = lambda *s, scale=1.0: jnp.asarray(rng.normal(0, scale, s).astype(np.float32))
    msa0, pair0 = arr(1, n, cm), arr(n, n, cp)
    w_opm, w_row = arr(cm, cp, scale=0.05), arr(cp, cm, scale=0.05)
    w_tri, w_single = arr(cp, cp, scale=0.05), arr(cm, cs, scale=0.05)

    opm = lambda m: (m[0] @ w_opm)[:, None, :] + (m[0] @ w_opm)[None, :, :]
    attn = lambda m, p: jnp.tanh(m + (p.mean(1) @ w_row)[None])
    tri = lambda p: jnp.tanh(p @ w_tri)

    def block(msa_0, pair_0, opm_first, msa_perturb):
        """msa_perturb is injected AT the msa_1 node, so its effect on pair_1 is ancestry."""
        if opm_first:
            pair_a = pair_0 + opm(msa_0)
            msa_1 = attn(msa_0, pair_a) + msa_perturb
            pair_1 = tri(pair_a)
        else:
            msa_1 = attn(msa_0, pair_0) + msa_perturb
            pair_a = pair_0 + opm(msa_1)
            pair_1 = tri(pair_a)
        return msa_1, pair_1

    def tail(single, pair):
        return jnp.sum(jnp.tanh(single)) * 1.7 + jnp.sum(jnp.tanh(pair)) * 0.9

    report = {"n": n}
    for opm_first in (True, False):
        variant = "multimer_opm_first" if opm_first else "monomer_opm_last"
        zero = jnp.zeros((1, n, cm), dtype=jnp.float32)

        # Ancestry: does a tangent at msa_1 reach pair_1?
        _, tangent = jax.jvp(
            lambda e: block(msa0, pair0, opm_first, e)[1], (zero,), (jnp.ones_like(zero),))
        reaches = float(jnp.linalg.norm(tangent))

        def whole(m0, p0):
            msa_1, pair_1 = block(m0, p0, opm_first, zero)
            return tail(msa_1[0] @ w_single, pair_1)

        truth = jax.grad(whole, argnums=(0, 1))(msa0, pair0)
        outputs = lambda m0, p0: block(m0, p0, opm_first, zero)
        out_val, vjp_fn = jax.vjp(outputs, msa0, pair0)
        seeds = (jax.grad(lambda m: tail(m[0] @ w_single, out_val[1]))(out_val[0]),
                 jax.grad(lambda p: tail(out_val[0][0] @ w_single, p))(out_val[1]))
        joint = vjp_fn(seeds)
        rel = lambda a, b: float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-12))
        report[variant] = {
            "tangent_at_msa_1_reaching_pair_1": reaches,
            "pair_1_descends_from_msa_1": reaches > 1e-9,
            "is_graph_cut": reaches <= 1e-9,
            "joint_injection_vs_truth_msa0": rel(joint[0], truth[0]),
            "joint_injection_vs_truth_pair0": rel(joint[1], truth[1]),
        }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


def cmd_seam(args):
    """The cotangent hand-off, measured on the real tail.

    The trunk reports four outputs, but they are not a graph cut. In EmbeddingsAndEvoformer
    (modules_multimer.py:440-452):

        single         = Linear(msa_activations[0])   <- 'evoformer/single_activations', a TRUNK
                                                         parameter, so the Linear is device-side
        msa            = msa_activations[:num_msa]
        msa_first_row  = msa_activations[0]
        pair           = pair_activations

    Three of the four descend from msa_activations, and `single` descends from the same row that
    `msa_first_row` is. The real cut is (msa_activations, pair_activations) -- two tensors.

    So a device tape seeded only at (msa, pair) silently drops dL/dsingle. This measures what
    that costs: the share of dL/dmsa_activations[0] that arrives through the single path.
    """
    cfg = model_config()
    trunk, tail = load_params()
    batch, meta = build_state(n_target=args.n_target, n_binder=args.n_binder,
                              n_binder_real=args.n_binder_real)
    reps = random_representations(meta["n"], cfg)
    losses, _ = build_loss_set()
    key = jax.random.PRNGKey(0)

    program = tail_program(cfg, losses, batch, meta, tail, key)
    value, grads = jax.jit(jax.value_and_grad(program))(reps)

    w_single = trunk["alphafold/alphafold_iteration/evoformer/single_activations"]["weights"]
    via_single = grads["single"] @ w_single.T          # (n, msa_channel)
    via_msa = grads["msa"][0]
    via_first_row = grads["msa_first_row"]
    total_row0 = via_single + via_msa + via_first_row

    nrm = lambda x: float(jnp.linalg.norm(x.astype(jnp.float32)))
    report = {
        "n": meta["n"], "loss_value": float(value),
        "single_activations_weight_shape": list(w_single.shape),
        "single_activations_scope": "alphafold/alphafold_iteration/evoformer/single_activations"
                                    "  (TRUNK side -- the Linear is on the device)",
        "norms_into_msa_activations_row0": {
            "via_single": nrm(via_single),
            "via_msa": nrm(via_msa),
            "via_msa_first_row": nrm(via_first_row),
            "total": nrm(total_row0),
        },
        "share_of_row0_cotangent_via_single": nrm(via_single) / nrm(total_row0),
        "relative_error_if_single_path_dropped":
            nrm(total_row0 - (via_msa + via_first_row)) / nrm(total_row0),
        "cotangent_norms": {k: nrm(v) for k, v in sorted(grads.items())},
        "bytes_bf16_per_step_one_way": {
            "pair": int(np.prod(grads["pair"].shape)) * 2,
            "single": int(np.prod(grads["single"].shape)) * 2,
            "msa": int(np.prod(grads["msa"].shape)) * 2,
            "msa_first_row": int(np.prod(grads["msa_first_row"].shape)) * 2,
        },
    }
    report["bytes_bf16_round_trip_total"] = 2 * sum(
        report["bytes_bf16_per_step_one_way"].values())
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bind"); b.set_defaults(fn=cmd_bind)
    t = sub.add_parser("time")
    t.add_argument("--n-target", type=int, default=115)
    t.add_argument("--n-binder", type=int, default=141)
    t.add_argument("--n-binder-real", type=int, default=130)
    t.add_argument("--repeats", type=int, default=7)
    t.add_argument("--out")
    t.set_defaults(fn=cmd_time)
    m = sub.add_parser("seam")
    m.add_argument("--n-target", type=int, default=115)
    m.add_argument("--n-binder", type=int, default=141)
    m.add_argument("--n-binder-real", type=int, default=130)
    m.add_argument("--out")
    m.set_defaults(fn=cmd_seam)
    c = sub.add_parser("cut")
    c.add_argument("--n", type=int, default=256)
    c.add_argument("--out")
    c.set_defaults(fn=cmd_cut)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

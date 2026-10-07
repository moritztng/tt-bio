#!/usr/bin/env python3
"""Issue #21 end to end on CPU: what a validation fold scores, with and without the splice.

The reporter's signature is two numbers on a target they supplied: `Target_pLDDT` collapsed to
~0.29 and constant across ten different MPNN sequences, and `Interface_Residues` exactly equal to
the binder length. `bci-triage` reproduced that on CPU and located the cause; this harness is the
graded version, and it answers the one question the routing test cannot: do the reporter's own
metrics come back.

Three arms, one target, one binder sequence, no card and no device anywhere:

  pure      BindCraft 2 with nothing spliced. The program their L40S control ran, and the number
            every other arm has to return to.
  spliced   `evoformer_on_device(evo, extra)` installed and the fold opened inside `evo.on_host`,
            which is exactly what a campaign's validation ensemble gets. Run against a PRE-FIX
            tt_bio tree this is issue #21; run against a fixed one it must equal `pure`.

`--ttbio` picks which tt_bio tree supplies the splice, so the before and after arms are the real
two commits rather than a flag inside one of them. The device extra-MSA stack is stood in with an
identity map over the pair representation, as triage did: CPU cannot run the card's stack, and the
point is that a *different* stack on the *design* pool's weights wrecks the target, not which one.

`pure` and a fixed `spliced` must agree EXACTLY, not merely closely. After the fix a host fold runs
BindCraft 2's own program on BindCraft 2's own weights, so any residual difference means the
stand-down is still incomplete.
"""
import argparse
import json
import os
import platform
import sys
import time


def build_states(pdb, chain, binder_length, seed):
    """Target from a real structure, binder a fixed designed sequence. Both chains named as
    BindCraft 2's filters expect (`binder`/`target`)."""
    import jax.numpy as jnp
    from bindcraft.protein import Protein, ResidueFlags

    chains = Protein.from_structure(pdb, chains=chain)
    target = chains[chain] if chain in chains else next(iter(chains.values()))

    rng = _deterministic(binder_length, seed)
    binder = Protein.from_fasta(">A\n" + rng)
    binder = binder.replace(flags=jnp.full((len(binder),), int(ResidueFlags.DESIGN),
                                           dtype=jnp.uint8))
    return {"complex": {"binder": binder, "target": target},
            "binder_alone": {"binder": binder}}


def _deterministic(length, salt):
    """A fixed binder sequence. Not a designed one -- see the note on i_pTM in the report."""
    import hashlib
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    out = []
    digest = hashlib.sha256(f"bci21-{salt}".encode()).digest()
    while len(out) < length:
        for byte in digest:
            out.append(alphabet[byte % len(alphabet)])
            if len(out) == length:
                break
        digest = hashlib.sha256(digest).digest()
    return "".join(out)


class _Trunk:
    extra_blocks = 4


class _Pool:
    def __init__(self):
        self.current = _Trunk()
        self.used = []

    def use(self, name):
        self.used.append(name)


class _IdentityExtraMsa:
    """Stands in for `ExtraMsaOnDevice`: a DIFFERENT extra-MSA stack, which is the whole point."""

    def __init__(self):
        self.pool = _Pool()
        self.swapped = []
        self.built = 0

    def as_jax(self, slot):
        self.built += 1
        return lambda pair, msa_mask, pair_mask: pair


def metrics(states, predictions, binder_length):
    from bindcraft.filters import REGISTERED_FILTER_METRICS as REG

    out = {}
    for name in ("Target_pLDDT", "pLDDT", "pTM", "i_pTM", "Interface_Residues", "Binder_RMSD"):
        function = REG.get(name)
        if function is None:
            continue
        try:
            value = function(states, predictions)
        except Exception as error:                        # a metric that needs a state we did
            out[name] = f"ERR:{type(error).__name__}"     # not predict says so, it does not stop
            continue
        out[name] = None if value is None else float(value)
    out["binder_length"] = binder_length
    return out


def run(args):
    sys.path.insert(0, args.ttbio)
    import jax
    from bindcraft.af2 import AlphaFoldDesignModel
    from tt_bio import bindcraft2

    states = build_states(args.pdb, args.chain, args.binder, args.seed)
    binder_length = len(states["complex"]["binder"])
    target_length = len(states["complex"]["target"])
    sequence_parameters = dict(softmax_weight=1.0, one_hot_weight=1.0, temperature=0.01,
                               logit_scale=2.0)

    def fold():
        # Built INSIDE the arm's context: `RunModel.apply` is a `jax.jit` and the layer_stack
        # factory is consulted at trace time, so a model shared across arms would carry the first
        # arm's jaxpr into the second (`perf/bcx_extrawire/round_ab.py` documents this trap).
        model = AlphaFoldDesignModel(presets=args.preset, data_dir=args.params,
                                     num_recycle=args.recycle, dropout=False,
                                     length_bucket_size=args.bucket)
        return model.predict(states, **sequence_parameters)

    started = time.time()
    extra = _IdentityExtraMsa()
    if args.arm == "pure":
        predictions = fold()
    else:
        evo = bindcraft2.EvoformerOnDevice(_Pool(), blocks=48)
        with bindcraft2.evoformer_on_device(evo, extra):
            with evo.on_host("validation_model"):
                predictions = fold()

    result = metrics(states, predictions, binder_length)
    result.update({
        "arm": args.arm, "ttbio": args.ttbio, "preset": args.preset,
        "target": os.path.basename(args.pdb), "chain": args.chain,
        "target_length": target_length, "binder_length": binder_length,
        "recycle": args.recycle, "bucket": args.bucket, "seed": args.seed,
        "device_extra_msa_stacks_built": extra.built,
        "trunk_selected_for_this_fold": extra.pool.used,
        "seconds": round(time.time() - started, 2),
        "host": platform.node(), "jax": jax.__version__,
        "jax_platform": os.environ.get("JAX_PLATFORMS", ""),
        "loadavg": round(os.getloadavg()[0], 2),
    })
    print(json.dumps(result), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("pure", "spliced"), required=True)
    parser.add_argument("--ttbio", required=True,
                        help="tt_bio tree the splice comes from: the fixed worktree or a pre-fix "
                             "checkout. This is what makes before/after two commits, not a flag.")
    parser.add_argument("--pdb", default="/home/ttuser/bci_struct/4zqk.pdb")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    parser.add_argument("--preset", default="model_1_ptm")
    parser.add_argument("--binder", type=int, default=42)
    parser.add_argument("--recycle", type=int, default=3)
    parser.add_argument("--bucket", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

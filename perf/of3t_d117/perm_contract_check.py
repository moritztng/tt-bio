"""D117: the ref_space_uid_to_perm batch contract, its two failure modes, and the fix.

Both failure modes are produced with upstream's OWN functions on a batch built by
upstream's OWN collator, so nothing here is a mock of the defect:

  A  KeyError   -- a transform pops the key off the caller's dict and restores it onto
                   the new dict the map returned, so a second pass finds it gone.
                   Reproduced with openfold3's `reshape_per_sample_inputs`, which is
                   what `safe_multi_chain_permutation_alignment` calls, and which is
                   the same pattern as `model.py`'s sample-dim expansion.
  B  IndexError -- a tensor tree map descends into the mapping, so the batch value stops
                   being a list of per-sample dicts and becomes a dict of stacked
                   tensors. Reproduced with openfold3's `dict_multimap`, which is what
                   `openfold_batch_collator` calls once the special case is removed,
                   then consumed by the real per-sample split and the real uid lookup.

Then the same two transforms via tt_bio.openfold3_batch, and `check_perm_contract`
on every arm, showing it flags both before the alignment can swallow them.

  PYTHONPATH=<of3pkg>:<worktree> python3 perm_contract_check.py
"""

import inspect
import json
import os
import sys
import traceback

import torch

from openfold3.core.utils.permutation_alignment import reshape_per_sample_inputs
from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map

from tt_bio.openfold3_batch import PERM_KEY, check_perm_contract, map_batch_tensors


def load_upstream_collator():
    """Pull `openfold_batch_collator` verbatim out of the installed data_module.py.

    Importing that module drags in pytorch_lightning and lmdb, which this box does not
    have and which have nothing to do with collation. Taking the function's own source
    out of the installed file keeps the evidence upstream's rather than a paraphrase:
    the path, line range and sha256 of the extracted text are reported alongside it.
    """
    import ast
    import hashlib

    import openfold3

    path = os.path.join(
        os.path.dirname(openfold3.__file__),
        "core", "data", "framework", "data_module.py",
    )
    text = open(path).read()
    tree = ast.parse(text)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "openfold_batch_collator"
    )
    src = ast.get_source_segment(text, fn)
    ns = {"torch": torch, "dict_multimap": dict_multimap}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    provenance = {
        "file": path,
        "lines": f"{fn.lineno}-{fn.end_lineno}",
        "sha256": hashlib.sha256(src.encode()).hexdigest()[:16],
    }
    return ns["openfold_batch_collator"], provenance


openfold_batch_collator, COLLATOR_PROVENANCE = load_upstream_collator()

N_ATOM, N_TOKEN, N_UID = 12, 6, 3


def make_sample(seed: int) -> dict:
    """One sample with THREE ref spaces, the smallest batch that shows defect B.

    With a single ref space the recursed mapping is still indexable at uid 0 and the
    run looks fine, which is why the defect survived: it needs uid >= 1 to bite.
    """
    g = torch.Generator().manual_seed(seed)
    uid = torch.arange(N_UID, dtype=torch.int32).repeat_interleave(N_ATOM // N_UID)
    return {
        "ref_space_uid": uid,
        "residue_index": torch.arange(N_TOKEN, dtype=torch.int32),
        "atom_positions": torch.randn(N_ATOM, 3, generator=g, dtype=torch.float64),
        # uid u has u+1 symmetry-equivalent permutations of its own 4 atoms: ragged on
        # both axes, which is exactly why the feature cannot be stacked.
        PERM_KEY: {
            u: torch.stack([torch.arange(N_ATOM // N_UID, dtype=torch.int32)] * (u + 1))
            for u in range(N_UID)
        },
    }


def per_sample_lookup(batch, batch_size):
    """upstream's own consumption path, the two lines that raise.

    `multi_chain_permutation_alignment` splits the batch per sample with `value[i]`
    (permutation_alignment.py:1556) and `get_final_atom_permutation_index` looks a
    conformer up with `[ref_space_uid.item()]` (:1175).
    """
    for i in range(batch_size):
        mapping = batch[PERM_KEY][i]
        for uid in batch["ref_space_uid"][i].flatten().unique():
            perms = mapping[uid.item()]
            # :1178 branches on this, so a phantom axis changes the algorithm's answer
            assert perms.ndim == 2, f"uid {int(uid)} perms have shape {tuple(perms.shape)}"


def contract_verdict(batch, batch_size, required=True):
    try:
        check_perm_contract(batch, batch_size=batch_size, required=required)
        return "pass"
    except ValueError as e:
        return f"FLAGGED: {e}"


def run(name, description, build, consume):
    """One arm: build a batch, consume it the way upstream does, record both verdicts."""
    row = {"arm": name, "what": description}
    try:
        batch, batch_size = build()
        row["contract_check"] = contract_verdict(batch, batch_size)
        consume(batch, batch_size)
        row["outcome"] = "COMPLETED"
        row["error"] = None
    except Exception as e:
        row["outcome"] = "FELL BACK"
        row["error"] = f"{type(e).__name__}: {e}"
        row.setdefault("contract_check", "not reached")
        row["traceback_tail"] = traceback.format_exc().strip().splitlines()[-3:]
    return row


def collate_upstream(samples):
    return openfold_batch_collator([dict(s) for s in samples])


def collate_recursed(samples):
    """openfold_batch_collator with its ref_space_uid_to_perm special case removed.

    This is `collate1` as of3t-bondcov wrote it: a generic collate over the feature
    dict, with nothing saying one entry is not a tensor tree.
    """
    def pad(values):
        return torch.nn.utils.rnn.pad_sequence(
            values, batch_first=True, padding_value=0
        ).squeeze(-1)

    return dict_multimap(pad, [dict(s) for s in samples])


def main():
    samples = [make_sample(0), make_sample(1)]
    bs = len(samples)
    rows = []

    # ---- the contract itself, read off upstream's collator ---------------------
    ref = collate_upstream(samples)
    contract = {
        "value_type": type(ref[PERM_KEY]).__name__,
        "len": len(ref[PERM_KEY]),
        "batch_size": bs,
        "entry_type": type(ref[PERM_KEY][0]).__name__,
        "entry_keys": sorted(ref[PERM_KEY][0]),
        "entry_key_type": type(next(iter(ref[PERM_KEY][0]))).__name__,
        "perm_shapes": {u: tuple(t.shape) for u, t in sorted(ref[PERM_KEY][0].items())},
        "ref_space_uid_shape": tuple(ref["ref_space_uid"].shape),
    }

    # ---- POSITIVE: the contract held ------------------------------------------
    rows.append(run(
        "P_upstream_collate",
        "upstream collator, consumed once",
        lambda: (collate_upstream(samples), bs),
        per_sample_lookup,
    ))

    # ---- A: the pop strips the caller's dict ----------------------------------
    def map_sample_dim_upstream(batch):
        """model.py:674-676 verbatim: pop off the caller's dict, map, restore on the new one."""
        perm = batch.pop(PERM_KEY, None)
        out = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        if perm is not None:
            out[PERM_KEY] = perm
        return out

    def build_a():
        # The caller holds one feature dict and runs two forwards over it, which is the
        # configuration of3t-reference was in. Each forward keeps its own expanded copy;
        # the caller's dict is what carries into the next one.
        caller_batch = collate_upstream(samples)
        map_sample_dim_upstream(caller_batch)           # forward 1
        return map_sample_dim_upstream(caller_batch), bs  # forward 2

    rows.append(run(
        "A_pop_then_second_forward",
        "model.py:674-676 sample-dim expansion, two forwards over one caller dict",
        build_a,
        per_sample_lookup,
    ))

    # A as upstream's own function does it, no hand-written pattern at all
    def build_a_upstream():
        batch = collate_upstream(samples)
        batch = map_batch_tensors(batch, lambda t: t.unsqueeze(1))
        pred = torch.randn(bs, 2, N_ATOM, 3, dtype=torch.float64)
        reshape_per_sample_inputs(batch=batch, atom_positions_predicted=pred)
        out, _ = reshape_per_sample_inputs(batch=batch, atom_positions_predicted=pred)
        return out, bs * 2

    rows.append(run(
        "A2_reshape_per_sample_inputs_twice",
        "openfold3's own reshape_per_sample_inputs called twice on one batch",
        build_a_upstream,
        per_sample_lookup,
    ))

    # ---- B: the tree map descends into the mapping ----------------------------
    rows.append(run(
        "B_collate_recursed",
        "collate with the special case removed, so dict_multimap recurses",
        lambda: (collate_recursed(samples), bs),
        per_sample_lookup,
    ))

    # ---- the fix: one helper, both transforms ---------------------------------
    def build_fixed_a():
        batch = collate_upstream(samples)
        once = map_batch_tensors(batch, lambda t: t.unsqueeze(1))
        assert PERM_KEY in batch, "map_batch_tensors mutated the caller's dict"
        twice = map_batch_tensors(once, lambda t: t.unsqueeze(1))
        return twice, bs

    rows.append(run(
        "FIX_A_map_batch_tensors_twice",
        "tt_bio.openfold3_batch.map_batch_tensors applied twice to one dict",
        build_fixed_a,
        per_sample_lookup,
    ))

    def build_fixed_a2():
        # The tt-bio rule for a mutating upstream entry point: hand it a fresh mapped
        # dict every time. map_batch_tensors never touches the dict the caller keeps,
        # so the second forward is as well fed as the first.
        base = collate_upstream(samples)
        pred = torch.randn(bs, 2, N_ATOM, 3, dtype=torch.float64)
        for _ in range(2):
            batch = map_batch_tensors(base, lambda t: t.unsqueeze(1))
            out, _ = reshape_per_sample_inputs(
                batch=batch, atom_positions_predicted=pred
            )
        return out, bs * 2

    rows.append(run(
        "FIX_A2_fresh_mapped_dict_per_forward",
        "map_batch_tensors before each reshape_per_sample_inputs, twice",
        build_fixed_a2,
        per_sample_lookup,
    ))

    from tt_bio.openfold3_batch import collate_list_axis

    def build_fixed_b():
        stripped, carried = collate_list_axis(samples)
        def pad(values):
            return torch.nn.utils.rnn.pad_sequence(
                values, batch_first=True, padding_value=0
            ).squeeze(-1)
        batch = dict_multimap(pad, stripped)
        batch.update(carried)
        assert PERM_KEY in samples[0], "collate_list_axis mutated its inputs"
        return batch, bs

    rows.append(run(
        "FIX_B_collate_list_axis",
        "tt_bio.openfold3_batch.collate_list_axis then the generic stacking collate",
        build_fixed_b,
        per_sample_lookup,
    ))

    # ---- inference must not regress -------------------------------------------
    # Every batch the repo's own harnesses build is filtered to tensors, and inference
    # never carries the mapping at all. On such a batch map_batch_tensors must be the
    # tensor tree map it replaced, bit for bit.
    plain = {k: v for k, v in collate_upstream(samples).items() if torch.is_tensor(v)}
    via_helper = map_batch_tensors(plain, lambda t: t.unsqueeze(1))
    via_upstream = tensor_tree_map(lambda t: t.unsqueeze(1), plain)
    equivalence = {
        "keys_match": sorted(via_helper) == sorted(via_upstream),
        "bit_exact": all(
            torch.equal(via_helper[k], via_upstream[k]) for k in via_upstream
        ),
        "shapes": {k: tuple(v.shape) for k, v in sorted(via_helper.items())},
    }

    # ---- no sixth copy of the workaround --------------------------------------
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    owner = os.path.join("tt_bio", "openfold3_batch.py")
    copies = []
    for sub in ("tt_bio", "scripts", "perf"):
        for root, _, files in os.walk(os.path.join(repo, sub)):
            if "_vendor" in root:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, repo)
                if rel == owner or rel == os.path.relpath(__file__, repo):
                    continue
                if f'pop("{PERM_KEY}"' in open(full).read():
                    copies.append(rel)

    data_py = os.path.join(repo, "tt_bio", "openfold3_data.py")
    guards = {
        "workaround_copies_outside_owner": copies,
        "inference_disables_the_feature": (
            "add_ref_space_uid_to_perm=False" in open(data_py).read()
        ),
    }

    # ---- negative control for the check itself --------------------------------
    good = collate_upstream(samples)
    dropped = {k: v for k, v in good.items() if k != PERM_KEY}
    control = {
        "check_on_valid_batch": contract_verdict(good, bs),
        "check_on_dropped_key": contract_verdict(dropped, bs),
        "check_on_recursed": contract_verdict(collate_recursed(samples), bs),
        "check_ignores_absent_key_when_not_required": contract_verdict(
            dropped, bs, required=False
        ),
    }

    import openfold3
    out = {
        "upstream": {
            "openfold3": os.path.dirname(openfold3.__file__),
            "collator": COLLATOR_PROVENANCE,
            "reshape_per_sample_inputs": (
                f"{inspect.getsourcefile(reshape_per_sample_inputs)}:"
                f"{inspect.getsourcelines(reshape_per_sample_inputs)[1]}"
            ),
            "torch": torch.__version__,
        },
        "contract": contract,
        "arms": rows,
        "equivalence": equivalence,
        "guards": guards,
        "check_control": control,
    }
    print(json.dumps(out, indent=2, default=str))

    failed = [r["arm"] for r in rows if r["arm"].startswith(("P_", "FIX_"))
              and r["outcome"] != "COMPLETED"]
    broke = [r["arm"] for r in rows if r["arm"].startswith(("A", "B"))
             and r["outcome"] != "FELL BACK"]
    bad_check = [k for k, v in control.items()
                 if ("dropped" in k or "recursed" in k) and not v.startswith("FLAGGED")
                 and "ignores" not in k]
    if control["check_on_valid_batch"] != "pass":
        bad_check.append("check_on_valid_batch")
    if control["check_ignores_absent_key_when_not_required"] != "pass":
        bad_check.append("check_ignores_absent_key_when_not_required")

    if not (equivalence["keys_match"] and equivalence["bit_exact"]):
        bad_check.append("equivalence_with_tensor_tree_map")
    if guards["workaround_copies_outside_owner"]:
        bad_check.append("workaround_copies_outside_owner")
    if not guards["inference_disables_the_feature"]:
        bad_check.append("inference_disables_the_feature")

    if failed or broke or bad_check:
        print(f"\nFAIL positives={failed} controls={broke} check={bad_check}",
              file=sys.stderr)
        return 1
    print("\nOK: 2 defects reproduced, 3 fixed arms complete, check flags both, "
          "helper bit-exact with tensor_tree_map, no workaround copies left",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

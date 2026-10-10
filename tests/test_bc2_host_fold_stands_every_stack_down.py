"""A fold routed to BindCraft 2's own trunk keeps BindCraft 2's WHOLE program, not just its Evoformer.

Issue #21. A campaign's validation ensemble is built with `trunk="jax"` and folds on the host, and
`docs/bindcraft2.md` and tt-bio's own "is not on card; folding it on BindCraft 2's own JAX trunk"
line both promise that. The promise was kept for the Evoformer alone. `predictor` installs the
extra-MSA and template swaps process-wide, and their chooser never consulted `host_only`, so a
validation fold kept BindCraft 2's Evoformer and lost its four extra-MSA blocks to the card's.

Those blocks then read their weights from `extra_msa.pool.current`, which is whichever checkpoint
the DESIGN loop last selected, because a host fold never reaches `pool.use`. Design is the multimer
pool, validation the monomer one, and both have four blocks so the count guard cannot fire. What a
reporter saw: `Target_pLDDT` ~0.29 constant across ten different sequences on a target they
supplied, and `Interface_Residues` exactly equal to the binder length, because a target whose pair
track has collapsed folds into the binder and every binder residue lands within 4 A of it.

So the contract this file pins is the routing one, not the metric: inside `on_host`, the splice
hands back BindCraft 2's own closure for EVERY stack it is asked for, and no device stack is built.
That is checkable in milliseconds with no card, no weights and no fold, which is the point -- the
end-to-end metric needs a chip and an hour, and this needs neither.
"""
import pytest

from tt_bio import bindcraft2

pytest.importorskip("bindcraft")


class _Trunk:
    extra_blocks = 4


class _Pool:
    """Stands in for `TrunkPool`: the splice only reads `current` and calls `use`."""

    def __init__(self):
        self.current = _Trunk()
        self.used = []

    def use(self, name):
        self.used.append(name)


class _Swap:
    """Counts the one thing that matters: was a device stack built for this fold?"""

    def __init__(self):
        self.pool = _Pool()
        self.swapped = []
        self.seen = {}
        self.built = 0

    def as_jax(self, slot):
        self.built += 1
        return lambda *a, **kw: a[0]


def _evo():
    return bindcraft2.EvoformerOnDevice(_Pool(), blocks=48)


def _extra_msa_stack_fn():
    """AlphaFold 2's extra-MSA per-block closure, as `find_extra_msa_masks` expects to find it.

    It reads `batch["use_dropout"]` as the real one does, which is where the splice takes it from.
    """
    extra_masks = {"msa": "MSA", "pair": "PAIR"}
    batch = {"use_dropout": True}

    def extra_msa_stack_fn(x):
        return extra_masks, batch["use_dropout"]

    return extra_msa_stack_fn


def _ask_the_splice_for_the_extra_msa_stack():
    from bindcraft.af.alphafold.model import modules
    return modules.layer_stack.layer_stack(4)(_extra_msa_stack_fn())


def test_a_card_fold_still_gets_the_cards_extra_msa_stack():
    """The control. Outside `on_host` nothing changes: this is what the design loop runs."""
    evo, extra = _evo(), _Swap()
    with bindcraft2.evoformer_on_device(evo, extra):
        _ask_the_splice_for_the_extra_msa_stack()
    assert extra.built == 1
    assert extra.swapped == [4]


def test_a_host_fold_keeps_bindcraft_2s_own_extra_msa_stack():
    """The regression. Before the fix `built` was 1 here: issue #21 in one assertion."""
    evo, extra = _evo(), _Swap()
    with bindcraft2.evoformer_on_device(evo, extra):
        with evo.on_host("validation_model"):
            _ask_the_splice_for_the_extra_msa_stack()
    assert extra.built == 0, "a host fold was handed the card's extra-MSA stack (issue #21)"
    assert extra.swapped == []


def test_the_host_fold_never_selects_a_trunk_so_it_must_not_use_one():
    """Why the wrong WEIGHTS followed from the wrong stack, pinned so it cannot come back.

    `_route` calls `pool.use(name)` only on the device arm. A host fold that still built a device
    stack therefore read `pool.current`, the design loop's last selection, and that is the multimer
    trunk standing in for the monomer validation one.
    """
    evo, extra = _evo(), _Swap()
    with bindcraft2.evoformer_on_device(evo, extra):
        with evo.on_host("validation_model"):
            _ask_the_splice_for_the_extra_msa_stack()
    assert extra.pool.used == [], "no trunk was selected for this fold, so none may be read"


def test_the_evoformer_still_stands_down_on_a_host_fold():
    """The behaviour that was already right, kept right by the reordering."""
    evo, extra = _evo(), _Swap()
    made = []
    with bindcraft2.evoformer_on_device(evo, extra):
        from bindcraft.af.alphafold.model import modules

        def evoformer_fn(x):
            return x

        with evo.on_host("validation_model"):
            made.append(modules.layer_stack.layer_stack(48)(evoformer_fn))
    assert evo.host_folds == {"validation_model": 1}


def _template_iteration_fn():
    import numpy as np
    padding_mask_2d = np.zeros((8, 8), dtype=np.float32)
    use_dropout = True

    def template_iteration_fn(carry):
        return padding_mask_2d, use_dropout

    return template_iteration_fn


def _drive_the_template_embedder(tmpl, evo, record):
    """Run the template embedder's splice without running AlphaFold's embedder.

    `template_on_device` swaps `modules_multimer.layer_stack` only while
    `SingleTemplateEmbedding.__call__` is on the stack, so the shim is unreachable from outside.
    Standing a recorder in for that method before the swap installs puts the test exactly where the
    real embedder sits, and keeps a 2-block template fold out of the test suite.
    """
    from bindcraft.af.alphafold.model import modules_multimer

    original = modules_multimer.SingleTemplateEmbedding.__call__

    def recorder(self, *args, **kwargs):
        record.append(modules_multimer.layer_stack.layer_stack(2)(_template_iteration_fn()))

    modules_multimer.SingleTemplateEmbedding.__call__ = recorder
    try:
        with bindcraft2.template_on_device(tmpl, evo):
            modules_multimer.SingleTemplateEmbedding.__call__(None)
    finally:
        modules_multimer.SingleTemplateEmbedding.__call__ = original


def test_a_card_fold_still_gets_the_cards_template_stack():
    """The control for the template embedder."""
    evo, tmpl, record = _evo(), _Swap(), []
    _drive_the_template_embedder(tmpl, evo, record)
    assert tmpl.built == 1
    assert tmpl.seen["blocks_swapped"] == 2


def test_a_host_fold_keeps_bindcraft_2s_own_template_stack():
    """The same hole in the template embedder, which bites when validation is on the multimer pool."""
    evo, tmpl, record = _evo(), _Swap(), []
    with evo.on_host("validation_model"):
        _drive_the_template_embedder(tmpl, evo, record)
    assert tmpl.built == 0, "a host fold was handed the card's template stack (issue #21)"
    assert "blocks_swapped" not in tmpl.seen

"""A block entry added for one site must not be inherited by another, tested without a device.

`_mm_block_at` is keyed on the weight's (kt, nt) and nothing else, and on top of that
`_mm_fused_block` DERIVES a key from the widths registered at a given `kt`. So an entry put in
`_MM_BLOCK` for one site is inherited by every site with a weight of that shape, and it also
widens the derived key space for every model.

`c12-diffusion-head-major` needs three entries for the AttentionPairBias and atom-block sites, and
(12, 48) is not merely a width a 16-head triangle attention at c_z=384 could have: it is ALREADY
LIVE as a derived key, because 12 + 36 is opendde's fused qkv+gate, which resolves to the swept
(4, 12, 1, 2, 1) and carries a measured 1.438 s. Putting an unswept `_MM_DEFAULT` there in the main
table replaces opendde's entry.

So the three live in `_MM_BLOCK_APB`, their own table, and `_mm_block_for_site` reads it only for
the sites that own it. This pins that: the table is scoped (every entry is `_MM_DEFAULT`, so no
swept tuning was hidden behind a site), it is invisible (the `triatt` path and every derived key
resolve exactly as they do without it), and it is reachable (the apb and atom sites get it).
"""
import sys
import pathlib

WT = str(pathlib.Path(__file__).resolve().parent.parent)
if WT not in sys.path:
    sys.path.insert(0, WT)


class W:
    dtype = None

    def __init__(self, k, n):
        self.shape = (k, n)


class X:
    """Enough of a tensor to reach the block lookup, which runs before any layout is read."""
    dtype = None

    def __init__(self, *dims):
        self.shape = dims
        self.padded_shape = dims


def test_apb_table_is_scoped_and_out_of_the_main_table():
    from tt_bio import tenstorrent as TT
    assert TT._MM_BLOCK_APB, "the table is empty, so it serves nothing"
    for key, blk in TT._MM_BLOCK_APB.items():
        assert blk is TT._MM_DEFAULT, \
            f"{key} is a SWEPT entry scoped to one site; a measured tuning must not be hidden there"
        assert key not in TT._MM_BLOCK, \
            f"{key} is in BOTH tables -- the main one wins and widens fused derivation with it"


def test_the_triatt_path_resolves_as_if_the_apb_table_did_not_exist():
    from tt_bio import tenstorrent as TT
    for kt, nt in TT._MM_BLOCK_APB:
        w = W(kt * 32, nt * 32)
        assert TT._mm_key(w) == (kt, nt)
        assert TT._mm_block_for_site(w, "triatt") is TT._mm_block_for(w)


def test_opendde_derived_fused_key_survives_at_12_48():
    """The regression this scoping exists for, named by its value rather than by its shape."""
    from tt_bio import tenstorrent as TT
    assert (12, 48) in TT._MM_BLOCK_APB
    swept = TT._MM_BLOCK[(12, 36)]
    assert swept is not TT._MM_DEFAULT
    assert TT._mm_fused_block(12, 48) == swept, "12 + 36 no longer derives opendde's qkv entry"
    assert TT._mm_block_for_site(W(384, 48 * 32), "triatt") == swept, \
        "opendde's fused tri-attention now resolves to an unswept _MM_DEFAULT"


def test_the_owning_sites_do_reach_the_table():
    from tt_bio import tenstorrent as TT
    for kt, nt in TT._MM_BLOCK_APB:
        for site in ("apb", "atom"):
            assert TT._mm_block_for_site(W(kt * 32, nt * 32), site) is TT._MM_DEFAULT, (kt, nt, site)


def test_the_atom_q_projection_still_rides_the_swept_gate_entry():
    """(4, 4) is the triangle attention's gate key and the atom q deliberately inherits it."""
    from tt_bio import tenstorrent as TT
    assert (4, 4) not in TT._MM_BLOCK_APB
    assert TT._mm_block_for_site(W(128, 128), "atom") is TT._MM_BLOCK[(4, 4)]


def test_negative_control_merging_the_tables_breaks_opendde():
    """A check that cannot be made to fail is not reading what it says it reads."""
    from tt_bio import tenstorrent as TT
    saved = dict(TT._MM_BLOCK)
    try:
        TT._MM_BLOCK.update(TT._MM_BLOCK_APB)
        assert TT._mm_block_for_site(W(384, 48 * 32), "triatt") is TT._MM_DEFAULT
    finally:
        TT._MM_BLOCK.clear()
        TT._MM_BLOCK.update(saved)
    assert TT._mm_block_for_site(W(384, 48 * 32), "triatt") == TT._MM_BLOCK[(12, 36)]


def test_a_swept_triatt_width_is_untouched():
    """opendde's own (12, 36) must serve at the triangle attention exactly as it ships."""
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_qkv as TQ
    assert TT._mm_key(W(384, 36 * 32)) == (12, 36)
    TQ.REJECTS.clear()
    # 12 heads at head_dim 32 is opendde. It must get past the block lookup and decline later, on
    # the stub's dtype, not on a site scope.
    assert TQ.qkv_heads(X(512, 512, 384), W(384, 12 * 3 * 32), None, 12, 32, None, None,
                        site="triatt") is None
    assert {r for r, _s in TQ.REJECTS} == {"dtype_or_memory"}, TQ.REJECTS


def test_the_apb_site_gets_past_the_block_lookup_at_its_own_width():
    from tt_bio import triatt_qkv as TQ
    was = TQ._APB_ENABLED
    TQ._APB_ENABLED = True
    TQ.APB_REJECTS.clear()
    try:
        # the diffusion token transformer's own width, whose key only `_MM_BLOCK_APB` carries
        assert TQ.qkv_heads(X(1, 512, 768), W(768, 3072), None, 16, 64, None, None,
                            site="apb") is None
        assert {r for r, _s in TQ.APB_REJECTS} == {"dtype_or_memory"}, TQ.APB_REJECTS
    finally:
        TQ._APB_ENABLED = was


def test_the_other_entry_points_refuse_mm_default_outright():
    """`gate_proj`, `qkvg_heads` and `qkvgb_heads` need no site scoping because of this.

    Read off the source rather than executed: each needs a live device past its guards, and the
    property being pinned is that the refusal exists at all.
    """
    src = (pathlib.Path(WT) / "tt_bio" / "triatt_qkv.py").read_text()
    for fn, needle in (("gate_proj", 'mm_default_entry_k1a_only'),
                       ("qkvg_heads", 'blk is None or blk is _MM_DEFAULT'),
                       ("qkvgb_heads", 'blk is None or blk is _MM_DEFAULT')):
        body = src[src.index(f"def {fn}("):]
        body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
        assert needle in body, f"{fn} no longer refuses a _MM_DEFAULT entry, so it now needs the " \
                               "site scoping that qkv_heads has"

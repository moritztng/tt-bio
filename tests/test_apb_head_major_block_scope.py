"""A block entry added for one site must not be inherited by another, tested without a device.

`_mm_block_for` is keyed on the weight's (kt, nt) and nothing else. `c12-diffusion-head-major` added
three entries for the AttentionPairBias and atom-block sites, and two of them are widths a TRIANGLE
attention could also have: `triatt_qkv.qkv_heads` accepts a `_MM_DEFAULT` entry (opendde depends on
that at (12, 36)) and its width condition is nt = 3 * n_heads at head_dim 32, so (12, 48) is a
16-head tri-attention at c_z=384 and (24, 96) a 32-head one at c_z=768. No fleet model has those
widths today, which is a coincidence and not a guard: the next port would have inherited an
unmeasured route with nothing to report it.

So `_MM_BLOCK_NOT_TRIATT` refuses them by key, and this pins all three halves of that:
the set is live (every key is in the table), it is scoped (every key is `_MM_DEFAULT`, so no swept
tuning was scoped away), and it bites (a tri-attention-shaped call at those keys declines with
`apb_only_block_entry` while the same shape at the APB site gets past it).
"""
import sys
import pathlib

WT = str(pathlib.Path(__file__).resolve().parent.parent)
if WT not in sys.path:
    sys.path.insert(0, WT)


class W:
    def __init__(self, k, n):
        self.shape = (k, n)


class X:
    """Enough of a tensor to reach the key guard, which runs before any dtype or layout is read."""
    def __init__(self, *dims):
        self.shape = dims
        self.padded_shape = dims


def test_guard_set_is_live_and_scoped():
    from tt_bio import tenstorrent as TT
    assert TT._MM_BLOCK_NOT_TRIATT, "the set is empty, so it guards nothing"
    for key in TT._MM_BLOCK_NOT_TRIATT:
        assert key in TT._MM_BLOCK, f"{key} is guarded but not in _MM_BLOCK -- a dead entry"
        assert TT._MM_BLOCK[key] is TT._MM_DEFAULT, \
            f"{key} is a SWEPT entry scoped away from the triangle attention; a measured tuning " \
            "must not be hidden behind a site guard"


def test_mm_key_matches_the_table_lookup():
    from tt_bio import tenstorrent as TT
    for k, n in ((768, 3072), (384, 1536), (128, 256), (128, 128), (256, 24 * 32)):
        assert TT._mm_key(W(k, n)) == (-(-k // 32), -(-n // 32))
        assert TT._MM_BLOCK.get(TT._mm_key(W(k, n))) is TT._mm_block_for(W(k, n))


def _triatt_call(TQ, kt_elems, n_heads):
    """A tri-attention qkv call at a width whose key is guarded: nt = 3 * n_heads at head_dim 32."""
    w = W(kt_elems, 3 * n_heads * 32)
    return TQ.qkv_heads(X(512, 512, kt_elems), w, None, n_heads, 32, None, None, site="triatt")


def test_triatt_declines_an_apb_only_entry_by_key():
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_qkv as TQ
    TQ.REJECTS.clear()
    before = list(TQ.STATS)
    # (12, 48) and (24, 96) are reachable by the tri-attention's own width condition.
    for kt_elems, n_heads in ((384, 16), (768, 32)):
        assert TT._mm_key(W(kt_elems, 3 * n_heads * 32)) in TT._MM_BLOCK_NOT_TRIATT
        assert _triatt_call(TQ, kt_elems, n_heads) is None
    reasons = {r for r, _shape in TQ.REJECTS}
    assert reasons == {"apb_only_block_entry"}, reasons
    assert TQ.STATS[1] == before[1] + 2 and TQ.STATS[0] == before[0], \
        "the decline must be counted as declined, not served"


def test_a_swept_triatt_width_still_serves_the_guard():
    """The guard must not fire on the widths that ship: opendde's (12, 36) is `_MM_DEFAULT` too."""
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_qkv as TQ
    assert TT._mm_key(W(384, 36 * 32)) == (12, 36)
    assert (12, 36) not in TT._MM_BLOCK_NOT_TRIATT
    TQ.REJECTS.clear()
    # 12 heads at head_dim 32 is opendde. It must get PAST the key guard and decline later, on the
    # stub's dtype, not on the site scope.
    assert TQ.qkv_heads(X(512, 512, 384), W(384, 12 * 3 * 32), None, 12, 32, None, None,
                        site="triatt") is None
    assert {r for r, _s in TQ.REJECTS} == {"dtype_or_memory"}, TQ.REJECTS


def test_apb_site_is_not_refused_by_the_key_guard():
    from tt_bio import triatt_qkv as TQ
    was = TQ._APB_ENABLED
    TQ._APB_ENABLED = True
    TQ.APB_REJECTS.clear()
    try:
        # the diffusion token transformer's own width, whose key IS in the guarded set
        assert TQ.qkv_heads(X(1, 512, 768), W(768, 3072), None, 16, 64, None, None,
                            site="apb") is None
        assert {r for r, _s in TQ.APB_REJECTS} == {"dtype_or_memory"}, TQ.APB_REJECTS
    finally:
        TQ._APB_ENABLED = was


def test_the_other_entry_points_refuse_mm_default_outright():
    """`gate_proj`, `qkvg_heads` and `qkvgb_heads` need no key guard because of this.

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
                               "key guard that qkv_heads has"

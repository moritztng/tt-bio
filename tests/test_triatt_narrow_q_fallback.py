"""`_tri_att_q_chunks` must offer a dividing q_chunk before one that pads.

WHY THIS EXISTS. `triatt_sdpa.sdpa` hoists the mask fill, and one of its preconditions is
`not use_padded_mask` -- so a q_chunk that does not DIVIDE the padded length does not merely pay
for the padding, it declines the fused path outright and drops the call to the stock op.

The ladder `_tri_att_q_chunks` returns is the dividing candidates wider than the production pick,
then the production pick itself, which is a fixed 256. So when L1 refuses every wide candidate,
the fused path survives only if 256 happens to divide the padded length.

MEASURED, rf3 on a Galaxy Wormhole (8x9 grid), size-ladder censuses:

    rung   ladder               256 divides?   TRIATT_PERSISTENT_MASK
     768   768, 384, 256              yes      1088 served,    0 fill_preconditions
     896   896, 448, 256               NO         0 served, 1088 fill_preconditions
    1024   1024, 512, 256             yes      1088 served,    0 fill_preconditions

896 refuses both wide entries (`pm_over_l1` 1087) and lands on a 256 that pads. 1024 refuses MORE
configs (`pm_over_l1` 2174) and serves anyway, because its fallback divides. The controlling
quantity is divisibility, not magnitude -- which is why this is a test and not a size threshold.

The fix offers the dividing chunks BELOW the production pick before falling back to it. It is off
by default (`TT_BIO_TRIATT_NARROW_Q_FALLBACK`): it changes which config a path shared by rf3,
boltz-2, protenix-v2, openfold3 and opendde picks at 13 of the 15 tile-aligned lengths from 640 to
1088, and one sequence length is not evidence for a default.

Device-free on purpose. It reads the function out of the source and runs its arithmetic against a
stub, so it cannot open a card and can run in any CPU job.
"""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "tt_bio" / "tenstorrent.py"
TILE = 32
PROD = 256          # _sdpa_chunks_shipped's pick at every length this test covers


def _ladder(padded: int, narrow_fallback: bool, prod: int = PROD) -> tuple:
    """Run the real `_tri_att_q_chunks` body against a stub, with no ttnn import."""
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_tri_att_q_chunks")
    fn.decorator_list = []                      # drop lru_cache, which would need the real module
    ns = {
        "SDPA_CHUNK_TILE": TILE,
        "_SDPA_WIDE_Q": True,
        "_SDPA_NARROW_Q_FALLBACK": narrow_fallback,
        "_sdpa_chunks_shipped": lambda q, k: (prod,),
        "_padded_sdpa_len": lambda q: padded,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_tri_att_q_chunks"](padded, padded)


ALIGNED = tuple(range(TILE, 1089, TILE))


@pytest.mark.parametrize("padded", ALIGNED)
def test_flag_off_is_byte_for_byte_the_shipped_ladder(padded):
    """Default-off must change nothing anywhere. This is the whole safety argument."""
    shipped = tuple(sorted((padded // n
                            for n in range(1, padded // TILE + 1)
                            if padded % n == 0 and (padded // n) % TILE == 0
                            and padded // n > PROD), reverse=True)) + (PROD,)
    assert _ladder(padded, narrow_fallback=False) == shipped


@pytest.mark.parametrize("padded", [n for n in ALIGNED if n % PROD == 0])
def test_a_length_whose_fallback_divides_is_unchanged_with_the_flag_on(padded):
    """768 and 1024 already work; turning the lever on must not perturb them at all."""
    assert _ladder(padded, narrow_fallback=True) == _ladder(padded, narrow_fallback=False)


@pytest.mark.parametrize("padded", [n for n in ALIGNED if n % PROD])
def test_a_length_whose_fallback_pads_gains_a_dividing_option_first(padded):
    """The point of the lever: reach a dividing chunk before the one that kills the fused path."""
    on = _ladder(padded, narrow_fallback=True)
    assert on[-1] == PROD, "the shipped fallback must stay last, not be removed"
    assert on[:-1] == tuple(sorted(on[:-1], reverse=True)), "widest-first: K and V are re-read per q chunk"
    assert all(padded % q == 0 for q in on[:-1]), "every offered chunk must divide the padded length"
    off = _ladder(padded, narrow_fallback=False)
    assert on[:len(off) - 1] == off[:-1], "the wide entries keep their order and priority"
    assert len(on) > len(off), f"{padded} gained no dividing option below {PROD}"


def test_the_896_case_this_was_root_caused_on():
    """The measured failure, pinned so a future ladder change cannot silently reintroduce it."""
    assert _ladder(896, narrow_fallback=False) == (896, 448, 256)
    assert _ladder(896, narrow_fallback=True) == (896, 448, 224, 128, 64, 32, 256)


def test_the_lever_is_off_by_default_in_the_source():
    """A shared-path lever does not land on by default off one sequence length's evidence."""
    tree = ast.parse(SRC.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "_SDPA_NARROW_Q_FALLBACK"):
            assert node.value.args[1].value is False
            return
    pytest.fail("_SDPA_NARROW_Q_FALLBACK is not assigned at module level in tenstorrent.py")

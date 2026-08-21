"""The accurate_softmax lever must stay off by default at every site that can reach it.

The lever trades perf for accuracy at a shared site, so a default flip has to be a deliberate,
measured, per-construction-site decision -- a Protenix-v2 fp32 default once cost OpenDDE 60x
(tt-bio-shared-diffusion-global-env-default-regression). This guards the no-op property by
assertion instead of by belief: every class that takes the keyword defaults it to False, and no
construction site in the shipped tree passes True.
"""
import inspect
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "tt_bio"

# Every class threading the keyword: the shared bf16 site, and the fp32 affinity trunk chain.
CLASSES = [
    "AttentionPairBias",
    "PairformerLayer",
    "Pairformer",
    "PairformerModule",
    "Fp32TriangleAttention",
    "Fp32AttentionPairBias",
    "Fp32PairformerLayer",
    "Fp32Pairformer",
    "Fp32PairformerModule",
]

# The A/B switches in boltz2.py, and the value each must read as when unset.
SWITCHES = ("TT_BIO_B2_ACCURATE_SOFTMAX", "TT_BIO_B2_AFFINITY_ACCURATE_SOFTMAX")


@pytest.mark.parametrize("name", CLASSES)
def test_keyword_exists_and_defaults_off(name):
    from tt_bio import tenstorrent

    cls = getattr(tenstorrent, name)
    params = inspect.signature(cls.__init__).parameters
    assert "accurate_softmax" in params, f"{name} lost the accurate_softmax keyword"
    assert params["accurate_softmax"].default is False, (
        f"{name} defaults accurate_softmax to {params['accurate_softmax'].default!r}; "
        "a flip needs its own measured accuracy AND perf number, per construction site"
    )


def test_no_construction_site_turns_it_on():
    """Pass-through (`accurate_softmax=accurate_softmax`) is fine; a literal True is not."""
    offenders = []
    for f in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"accurate_softmax\s*=\s*True", line):
                offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "accurate_softmax is enabled in the shipped tree:\n  " + "\n  ".join(offenders)
    )


def test_mask_floor_defaults_off_so_existing_callers_are_unchanged():
    """The floor fixes a fully-masked row but perturbs nothing else, and must stay opt-in.

    RF3's verdict rests on numbers measured without it, so a new default here would silently
    re-price a closed measurement.
    """
    from tt_bio.tenstorrent import _accurate_softmax

    params = inspect.signature(_accurate_softmax).parameters
    assert params["mask_floor"].default is None


def test_row_emptying_mask_site_passes_the_floor():
    """Fp32TriangleAttention's mask is [I,1,1,J] and empties whole rows.

    Without the floor the chain divides 0/0 there: ttnn.max rounds fp32 to bf16, so x - max is
    -1.76e6 rather than 0 on an all-masked row, exp underflows and the row sums to 0. Measured:
    nan on 7.67 M of 28.3 M elements at [192,4,192,192].
    """
    src = (SRC / "tenstorrent.py").read_text()
    body = src.split("class Fp32TriangleAttention")[1].split("\nclass ")[0]
    assert "_accurate_softmax(sc, ckc, mask_floor=_MASK_EXP_FLOOR)" in body, (
        "Fp32TriangleAttention must pass mask_floor: its pair mask empties whole rows"
    )


@pytest.mark.parametrize("var", SWITCHES)
def test_screen_switch_defaults_off(var, monkeypatch):
    """Unset must mean off, so a stray environment cannot re-price a fold."""
    monkeypatch.delenv(var, raising=False)
    src = (SRC / "boltz2.py").read_text()
    assert f'os.environ.get("{var}", "0") == "1"' in src, (
        f"{var} must default to off when unset"
    )


def test_both_boltz2_paths_are_reachable_by_a_switch():
    """All four PairformerModule sites plus the affinity Fp32PairformerModule take the keyword.

    A screen that only reaches some of a model's construction sites reads as a null result at the
    ones it misses (same shape as rfd3-tile-sparsity-and-wrong-variable-gate).
    """
    src = (SRC / "boltz2.py").read_text()
    assert src.count("accurate_softmax=accurate_sm_bf16") == 3
    assert src.count("accurate_softmax=accurate_sm_fp32") == 1
    # the affinity head's own stack reads the bf16 switch directly at its construction site
    assert src.count('"TT_BIO_B2_ACCURATE_SOFTMAX", "0") == "1"') == 2

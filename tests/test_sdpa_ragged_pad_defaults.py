"""The fused SDPA's ragged-tile-tail mask is per site, and RF3 is the one site it ships on.

ttnn's fused SDPA reduces over the TILE-PADDED physical key length while a mask sized to the
logical length leaves the ragged tail at a bias of 0. exp(0) = 1 beats real scores that mostly sit
below 0, so up to 31 padded key columns take a real share of every row's softmax mass. Measured in
fp64 on captured RF3 operands the fused path is 71-76x the materialised path's error at every
length that is not a multiple of 32, and flat at ~1e-2 either way when it is.

So the mask is a correctness fix, not a tuning knob -- and it is still per site, because scoring it
on the three models that reach a ragged fused call did not give one answer:

    rf3 7ROA L117          X 1.6415 A (outside floor)  ->  X 0.1780 A          9.2x BETTER
    protenix-v2 hsa L585   committed GAP               ->  X 0.674 / 0.695     GAP -> PASS
    protenix-v2 ubq L76    committed GAP-evidenced     ->  X 1.828 / 1.923     GAP -> PASS
    opendde-prot-prod      X 2.215 / 2.102 (PASS)      ->  X 7.289 / 3.683     PASS -> GAP
    opendde-trpcage-nomsa  X 0.361 / 0.374 (PASS)      ->  X 0.428 / 0.475     19% WORSE

OpenDDE also gets less self-consistent with the mask on (device floor 2.102 -> 3.683), the opposite
of what it does to RF3, so that is not diffusion chaos being misread. Until it is root-caused, the
site that measured a win is the only site that ships it. Protenix-v2 improving at both targets is
a follow-up worth taking, not something to fold in here.

`TT_BIO_SDPA_RAGGED_PAD` still forces the mask on at EVERY site; that global is how the table above
was measured and it stays as the cross-model A/B switch.

UPDATE 2026-08-24 -- THE GLOBAL NOW DEFAULTS ON, and the table above must be read with its date.
Every row of it was measured while those models ran a RAGGED token axis. They do not any more:
protenix-v2 and opendde bucket by default (1208 and 1216 ragged fused-SDPA calls -> 0), and every
one of the 18 shipped models buckets now. So the guard fires on NOTHING on any shipped path, which
is verified here and on hardware rather than argued -- `SDPA_RAGGED_PAD_STATS` reads 0 and the folds
are bit-identical with the global off and on.

That is the whole reason the OpenDDE regression did not block the flip: it was not explained away,
the INPUT that produced it no longer occurs. The per-site selector stays for the day a model loses
its bucket.

This is a CORRECTNESS GUARD, not a perf knob. Do not "simplify" it back to opt-in. An unmasked
ragged tail makes the softmax denominator sum garbage columns -- wrong math, the same class as
PLAYBOOKS §MODEL 2b's 72x finding -- and a fix you have to know to ask for protects only the people
who already knew. Measured at a genuinely ragged length it is free (-1.1 %, inside noise, because
the pad aliases in TILE layout) and it corrects the answer.
"""
import ast
import inspect
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tt_bio"

# Construction sites wired to the selector, with the file that wires them.
SITES = {"rf3.tri_att": "tt_bio/rf3/remap.py"}

# RF3 ships ON. Unlike the HiFi4 sibling this lever has a fold-level number behind it at this site,
# and shipping RF3 OFF is not neutral: the fast arm without the mask reads 1.6415 A at 7ROA against
# a 0.361 A floor, which is worse than the materialised route it replaced.
ON_BY_DEFAULT = {"rf3.tri_att"}


def _live():
    from tt_bio.tenstorrent import sdpa_ragged_pad_site
    return {s for s in SITES if sdpa_ragged_pad_site(s, default=s in ON_BY_DEFAULT)}


def test_env_unset_ships_rf3_on_and_nothing_else(monkeypatch):
    monkeypatch.delenv("TT_BIO_SDPA_RAGGED_PAD_AB", raising=False)
    assert _live() == ON_BY_DEFAULT


def test_empty_env_ships_rf3_on_and_nothing_else(monkeypatch):
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "")
    assert _live() == ON_BY_DEFAULT


@pytest.mark.parametrize("token", sorted(SITES))
def test_a_named_site_moves_only_itself(monkeypatch, token):
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-" + token)
    assert _live() == ON_BY_DEFAULT - {token}
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", token)
    assert _live() == ON_BY_DEFAULT | {token}


def test_all_and_minus_all_move_every_site_without_a_token_of_its_own(monkeypatch):
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "all")
    assert _live() == set(SITES)
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-all")
    assert _live() == set()
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-all,rf3.tri_att")
    assert _live() == {"rf3.tri_att"}


def test_the_three_selectors_do_not_move_each_other(monkeypatch):
    """All three share `_site_flag`, so the one thing that refactor can break is two of them
    reading each other's environment variable."""
    from tt_bio.tenstorrent import (accurate_softmax_site, sdpa_ragged_pad_site,
                                    triatt_sdpa_hifi_site)
    for var in ("TT_BIO_SDPA_RAGGED_PAD_AB", "TT_BIO_TRIATT_SDPA_HIFI_AB",
                "TT_BIO_ACCURATE_SOFTMAX_AB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "all")
    assert sdpa_ragged_pad_site("rf3.tri_att")
    assert not triatt_sdpa_hifi_site("boltz2.trunk")
    assert not accurate_softmax_site("openfold3.trunk")
    monkeypatch.setenv("TT_BIO_SDPA_RAGGED_PAD_AB", "-all")
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "all")
    assert triatt_sdpa_hifi_site("boltz2.trunk")
    assert not sdpa_ragged_pad_site("rf3.tri_att", default=True)


@pytest.mark.parametrize("token", sorted(SITES))
def test_site_is_wired_in_the_named_file(token):
    src = (ROOT / SITES[token]).read_text()
    assert 'sdpa_ragged_pad_site("%s"' % token in src, \
        "%s is not wired in %s" % (token, SITES[token])


def test_opendde_and_protenix_do_not_get_a_token():
    """The whole point of scoping. A token for either would ship the regression OpenDDE measured,
    or ship Protenix an improvement nobody signed off on."""
    offenders = []
    call = re.compile(r"sdpa_ragged_pad_site\(\s*f?\"([^\"]*)\"")
    for path in sorted(SRC.rglob("*.py")):
        offenders += [t for t in call.findall(path.read_text()) if t not in SITES]
    assert offenders == [], "unlisted sites wired to the pad selector: %s" % sorted(offenders)


def test_no_site_bypasses_the_selector_with_a_literal():
    """A flip moves a `default=`, it does not hardcode the keyword: an unnamed site cannot be
    forced back off, and being unable to turn this one OFF is what makes the OpenDDE reading
    unreproducible."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]  # prose that NAMES the keyword is not a bypass
            if re.search(r"(tri_att_)?sdpa_ragged_pad\s*=\s*True", code):
                offenders.append("%s:%d" % (path.relative_to(ROOT), i))
    assert offenders == [], "the mask is hardcoded ON at: %s" % ", ".join(offenders)


@pytest.mark.parametrize("cls_name,param", [
    ("TriangleAttention", "sdpa_ragged_pad"),
    ("PairformerLayer", "tri_att_sdpa_ragged_pad"),
    ("Pairformer", "tri_att_sdpa_ragged_pad"),
])
def test_every_layer_on_the_path_defaults_the_keyword_off(cls_name, param):
    """Every model that did not opt in must be byte-for-byte unaffected, which holds only if the
    keyword defaults off the whole way down."""
    import tt_bio.tenstorrent as T
    assert inspect.signature(getattr(T, cls_name).__init__).parameters[param].default is False


def test_pairformer_layer_routes_the_flag_to_both_triangle_attentions_and_nothing_else():
    """Read off the AST: building the layer would need weights and a card. Triangle attention has
    a start and an ending variant and a ragged length is ragged in both."""
    tree = ast.parse((ROOT / "tt_bio/tenstorrent.py").read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "PairformerLayer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    passed = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for kw in node.keywords:
                if kw.arg == "sdpa_ragged_pad":
                    passed.setdefault(node.func.id, []).append(ast.unparse(kw.value))
    assert passed == {"TriangleAttention": ["tri_att_sdpa_ragged_pad"] * 2}, \
        "the flag must reach both TriangleAttentions and nothing else: %s" % passed


def test_the_global_still_forces_every_site_on():
    """`TT_BIO_SDPA_RAGGED_PAD` is how the cross-model table in this module's docstring was
    measured. If the per-site flag replaced it rather than joining it, that measurement stops
    being reproducible."""
    src = (ROOT / "tt_bio/tenstorrent.py").read_text()
    assert "if not (ragged and (_SDPA_RAGGED_PAD or pad)):" in src


def test_rf3_ships_the_mask_on():
    """The regression guard for the half-applied flip. RF3 on the fused arm WITHOUT the mask is
    worse than the materialised route it replaced, so `fp32_softmax=False` and this flag are one
    change and must not drift apart."""
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS
    assert PAIRFORMER_FLAGS["fp32_softmax"] is False
    assert PAIRFORMER_FLAGS["tri_att_sdpa_ragged_pad"] is True


# ---------------------------------------------------------------------------------------------
# The global default. Moritz, 2026-08-24, ask 6378: flip it ON.
# ---------------------------------------------------------------------------------------------

def test_the_global_default_is_on_with_the_env_var_ABSENT():
    """Not "the env var works" -- the DEFAULT, read with nothing in the environment.

    `_SDPA_RAGGED_PAD` is bound at import, so monkeypatching this process cannot test it: the
    module was imported long before the test ran, and a `delenv` here would pass against a stale
    True. A child with the variable stripped from its environment is the only honest check, and it
    is the exact condition a user gets.
    """
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "TT_BIO_SDPA_RAGGED_PAD"}
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run(
        [sys.executable, "-c",
         "import tt_bio.tenstorrent as T; print('VALUE', T._SDPA_RAGGED_PAD)"],
        capture_output=True, text=True, env=env, timeout=600)
    assert "VALUE True" in out.stdout, (
        "the fused-SDPA ragged-tail mask must default ON; child said:\n"
        + out.stdout[-2000:] + out.stderr[-2000:])


def test_the_escape_hatch_still_turns_it_off():
    """Condition 4 of the flip: the old behaviour stays one env var away for bisecting."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, TT_BIO_SDPA_RAGGED_PAD="0", PYTHONPATH=str(ROOT))
    out = subprocess.run(
        [sys.executable, "-c",
         "import tt_bio.tenstorrent as T; print('VALUE', T._SDPA_RAGGED_PAD)"],
        capture_output=True, text=True, env=env, timeout=600)
    assert "VALUE False" in out.stdout, (
        "TT_BIO_SDPA_RAGGED_PAD=0 must restore the unmasked behaviour; child said:\n"
        + out.stdout[-2000:] + out.stderr[-2000:])


def test_the_default_is_a_literal_True_at_the_one_place_it_is_read():
    """Reading the source, so a refactor that moves the flag behind a helper still has to say
    True out loud rather than inherit a False from somewhere else."""
    src = (ROOT / "tt_bio/tenstorrent.py").read_text()
    assert '_SDPA_RAGGED_PAD = env_flag("TT_BIO_SDPA_RAGGED_PAD", True)' in src


def _device_available():
    import os
    return bool(os.environ.get("TT_VISIBLE_DEVICES")) and os.path.exists("/dev/tenstorrent")


@pytest.mark.skipif(not _device_available(),
                    reason="needs a TT card and TT_VISIBLE_DEVICES pinned to it")
def test_a_ragged_call_really_is_masked():
    """The guard's actual job, on hardware: at a ragged key length the padded columns come back
    carrying the -1e9, and the aligned case is left completely alone.

    Checks the bias `_sdpa_pad_ragged` hands the kernel rather than the kernel's output, because
    that is where the defect lives -- ttnn reduces over the TILE-PADDED key length, so the only
    thing standing between a real score and a padded column is what is in the bias there.
    """
    import torch
    import ttnn
    from tt_bio.tenstorrent import _sdpa_pad_ragged, _SDPA_PAD_MASK, get_device
    dev = get_device()
    B, H, S, D = 1, 2, 100, 32                     # 100 % 32 = 4 -> physically 128
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    q = up(torch.zeros(B, H, S, D)); k = up(torch.zeros(B, H, S, D))
    v = up(torch.zeros(B, H, S, D)); bias = up(torch.zeros(B, H, S, S))
    _q, _k, _v, bp, q_pad = _sdpa_pad_ragged(q, k, v, bias)
    assert q_pad == 28, q_pad
    got = ttnn.to_torch(bp).float()
    assert got.shape[-1] == 128, got.shape
    assert (got[..., :S, S:] <= _SDPA_PAD_MASK / 2).all(), \
        "padded KEY columns are not masked: max %r" % float(got[..., :S, S:].max())
    assert (got[..., :S, :S] == 0).all(), "the real region was disturbed"
    assert (got[..., S:, :S] == 0).all(), \
        "padded QUERY rows must pad with 0, not the mask -- a fully masked row divides by zero"

    # ...and an already-aligned call is not touched at all, which is why the default-ON flip
    # cannot move a number on any bucketed model.
    qa = up(torch.zeros(B, H, 128, D)); ba = up(torch.zeros(B, H, 128, 128))
    out = _sdpa_pad_ragged(qa, qa, qa, ba)
    assert out[4] == 0 and out[3] is ba, "an aligned call must be returned untouched"

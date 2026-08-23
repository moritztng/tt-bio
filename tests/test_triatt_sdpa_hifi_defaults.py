"""The fused triangle-attention SDPA fidelity lever is per-site, and it ships off everywhere.

The fused kernel takes its compute kernel config per call. Left alone it runs the op default
(HiFi2, math_approx on, no fp32_dest_acc), which scored worst of the eight configs in
`perf/rf3/triatt_ckc_sweep.py`. `triatt_sdpa_hifi_site` lets one construction site ask for
(HiFi4, approx off, fp32_dest_acc) instead.

Boltz-2's trunk is the only site anyone has scored at fold level, and the answer was FLAT: the
config is 1.98x lower total error per call in fp64, one-signed 18/18, and 64 residual+LayerNorm
blocks absorb it down to -1.81% on `z` at 512 aa, +0.003% at 320 aa, with `s` 5.96% worse at 512
(`~/.coworker/state/fused-sdpa-crossmodel-hifi4.md`). So nothing ships on, and the two properties
worth locking are that no unmeasured site joins by accident, and that the lever stays A/B-able
per site -- the selector exists so the Protenix-v2 and OpenDDE question can be re-opened once
their ragged-tile-tail error stops dominating it.

Per site, never `triatt_sdpa._CKC_OVERRIDE`: that module global is shared by six models, and a
switch two models share is the shape of the Protenix-v2 fp32 default that cost OpenDDE 60x.
"""
import ast
import inspect
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tt_bio"

# Every construction site wired to the selector, with the file that wires it. Boltz-2's 64-block
# trunk is the only one: the 8-block confidence head, the 2-block template stack and the affinity
# stack are unscored, so they do not get a token until something scores them.
SITES = {"boltz2.trunk": "tt_bio/boltz2.py"}

# Empty on purpose. Fold level said FLAT.
ON_BY_DEFAULT = set()


def _live():
    from tt_bio.tenstorrent import triatt_sdpa_hifi_site
    return {s for s in SITES if triatt_sdpa_hifi_site(s, default=s in ON_BY_DEFAULT)}


def test_env_unset_ships_every_site_off(monkeypatch):
    monkeypatch.delenv("TT_BIO_TRIATT_SDPA_HIFI_AB", raising=False)
    assert _live() == ON_BY_DEFAULT


def test_empty_env_ships_every_site_off(monkeypatch):
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "")
    assert _live() == ON_BY_DEFAULT


@pytest.mark.parametrize("token", sorted(SITES))
def test_a_named_site_turns_on_only_itself(monkeypatch, token):
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", token)
    assert _live() == ON_BY_DEFAULT | {token}
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "-" + token)
    assert _live() == ON_BY_DEFAULT - {token}


def test_all_and_minus_all_move_every_site_without_a_token_of_its_own(monkeypatch):
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "all")
    assert _live() == set(SITES)
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "-all")
    assert _live() == set()
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "-all,boltz2.trunk")
    assert _live() == {"boltz2.trunk"}


def test_the_two_levers_do_not_move_each_other(monkeypatch):
    """`accurate_softmax_site` and `triatt_sdpa_hifi_site` share `_site_flag`, so the one thing
    the refactor could have broken is the two reading each other's environment variable."""
    from tt_bio.tenstorrent import accurate_softmax_site, triatt_sdpa_hifi_site
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "all")
    monkeypatch.delenv("TT_BIO_ACCURATE_SOFTMAX_AB", raising=False)
    assert triatt_sdpa_hifi_site("boltz2.trunk") and not accurate_softmax_site("openfold3.trunk")
    monkeypatch.setenv("TT_BIO_ACCURATE_SOFTMAX_AB", "all")
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", "-all")
    assert accurate_softmax_site("openfold3.trunk") and not triatt_sdpa_hifi_site("boltz2.trunk")


@pytest.mark.parametrize("token", sorted(SITES))
def test_site_is_wired_in_the_named_file(monkeypatch, token):
    monkeypatch.setenv("TT_BIO_TRIATT_SDPA_HIFI_AB", token)
    from tt_bio.tenstorrent import triatt_sdpa_hifi_site
    assert triatt_sdpa_hifi_site(token), "%s is not reachable through the selector" % token
    src = (ROOT / SITES[token]).read_text()
    assert 'triatt_sdpa_hifi_site("%s")' % token in src, \
        "%s is not wired in %s" % (token, SITES[token])


def test_no_site_ships_the_lever_on():
    """FLAT at the only site anyone measured. A `default=True` here means a site shipped an
    arithmetic change to production without a fold-level number behind it."""
    on = set()
    call = re.compile(r"triatt_sdpa_hifi_site\(\s*(f?\"[^\"]*\")\s*,\s*default=True\s*\)")
    for path in sorted(SRC.rglob("*.py")):
        on |= set(call.findall(path.read_text()))
    assert on == set(), "sites shipping the fidelity lever on: %s" % sorted(on)


def test_no_site_bypasses_the_selector_with_a_literal():
    """A flip moves a `default=`, it does not hardcode the keyword. An unnamed site cannot be
    forced back off, which is how a shared default becomes a 60x."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"(tri_att_)?sdpa_hifi\s*=\s*True", line):
                offenders.append("%s:%d" % (path.relative_to(ROOT), i))
    assert offenders == [], "the lever is hardcoded ON at: %s" % ", ".join(offenders)


@pytest.mark.parametrize("cls_name,param", [
    ("TriangleAttention", "sdpa_hifi"),
    ("PairformerLayer", "tri_att_sdpa_hifi"),
    ("Pairformer", "tri_att_sdpa_hifi"),
    ("PairformerModule", "tri_att_sdpa_hifi"),
])
def test_every_layer_on_the_path_defaults_the_keyword_off(cls_name, param):
    """The off arm has to be a byte-for-byte no-op, and it is only that if every caller that does
    not opt in keeps `ckc=None` all the way to `triatt_sdpa.sdpa`."""
    import tt_bio.tenstorrent as T
    assert inspect.signature(getattr(T, cls_name).__init__).parameters[param].default is False


def test_pairformer_layer_routes_the_flag_to_both_triangle_attentions_and_nothing_else():
    """Read off the AST: building the layer would need weights and a card."""
    tree = ast.parse((ROOT / "tt_bio/tenstorrent.py").read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "PairformerLayer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    passed = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for kw in node.keywords:
                if kw.arg == "sdpa_hifi":
                    passed.setdefault(node.func.id, []).append(ast.unparse(kw.value))
    assert passed == {"TriangleAttention": ["tri_att_sdpa_hifi", "tri_att_sdpa_hifi"]}, \
        "the flag must reach both TriangleAttentions and nothing else: %s" % passed


def test_the_off_arm_asks_the_kernel_for_the_op_default():
    """`sdpa_hifi` False has to mean `ckc=None`, not some other tuple: `None` is what makes
    `triatt_sdpa.sdpa` fall through to the op default it used before this parameter existed."""
    src = (ROOT / "tt_bio/tenstorrent.py").read_text()
    assert "_TRIATT_FUSED_HIFI_CKC if self.sdpa_hifi else None" in src

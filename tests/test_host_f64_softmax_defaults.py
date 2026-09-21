"""The host float64 softmax is a per-site path, and it ships off at every site.

Tenstorrent's fp32 is a few mantissa bits short of IEEE fp32. On [1,16,384,384] fp32 against a
float64 softmax of the same values the op's own default reads 2.029e-02, `precise_config()`
1.646e-03 and the 5-op `_accurate_softmax` chain 5.156e-04, where real fp32 reads ~1e-7. Upstream
trains in IEEE fp32 on GPU, so a host round trip is not overshooting them, it is the only route to
what they already do. It costs a round trip per softmax and it is a training-path lever.

Three properties to hold. Nothing ships it on. Every site that owns a replaceable `ttnn.softmax`
has one, so the path cannot silently miss a site the precise lever already reaches. And with the
site off, `site_softmax` is `ttnn.softmax` and nothing else, which is what makes a fold with the
path present byte-identical to one without it.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tt_bio"

# Every construction site that resolves a softmax token, with the file that owns it. These are
# the sites whose `ttnn.softmax` this path can replace -- the same set `softmax_ckc` reaches,
# because the two levers act on the same call.
SITES = {
    "openfold3.diffusion_transformer": "tt_bio/openfold3_diffusion_transformer.py",
    "openfold3.atom_transformer": "tt_bio/openfold3_atom_transformer.py",
    "protenix.atom_transformer": "tt_bio/protenix.py",
}

ENV = "TT_BIO_HOST_F64_SOFTMAX_AB"


def _selector():
    from tt_bio.tenstorrent import host_f64_softmax_site
    return host_f64_softmax_site


def _live():
    sel = _selector()
    return {s for s in SITES if sel(s)}


def test_env_unset_ships_every_site_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _live() == set()


def test_empty_env_ships_every_site_off(monkeypatch):
    monkeypatch.setenv(ENV, "")
    assert _live() == set()


@pytest.mark.parametrize("token", sorted(SITES))
def test_a_named_site_turns_on_only_itself(monkeypatch, token):
    monkeypatch.setenv(ENV, token)
    assert _live() == {token}


def test_all_and_minus_all_move_every_site_without_a_token_of_its_own(monkeypatch):
    monkeypatch.setenv(ENV, "all")
    assert _live() == set(SITES)
    monkeypatch.setenv(ENV, "-all")
    assert _live() == set()
    monkeypatch.setenv(ENV, "-all,openfold3.atom_transformer")
    assert _live() == {"openfold3.atom_transformer"}


def test_no_site_ships_the_path_on():
    """A `default=True` here would put a host round trip into a user's fold. There is none, and
    a flip would have to move this line as well as the call site."""
    call = re.compile(r"host_f64_softmax_site\(\s*[^)]*default\s*=\s*True")
    offenders = [str(p.relative_to(ROOT)) for p in sorted(SRC.rglob("*.py"))
                 if call.search(p.read_text())]
    assert offenders == [], "the host float64 softmax ships on at: %s" % ", ".join(offenders)


def test_every_precise_softmax_site_also_has_a_float64_one():
    """The anti-drift check. `softmax_ckc(token)` and `host_f64_softmax_site(token)` decide the
    same call, so a site that gained one lever and not the other is a site this path misses."""
    ckc, f64 = set(), set()
    for path in sorted(SRC.rglob("*.py")):
        t = path.read_text()
        ckc |= set(re.findall(r"softmax_ckc\(\s*\"([^\"]+)\"", t))
        f64 |= set(re.findall(r"host_f64_softmax_site\(\s*\"([^\"]+)\"", t))
    assert ckc == f64, "sites with a precise lever but no float64 one: %s; the other way: %s" % (
        sorted(ckc - f64), sorted(f64 - ckc))
    assert ckc == set(SITES), "the site list in this file is stale: %s" % sorted(ckc)


def test_no_wired_site_still_calls_ttnn_softmax_directly():
    """Every call that takes `self._softmax_ckc` must go through `site_softmax`, or the site has
    a float64 selector that decides nothing."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "ttnn.softmax(" in line and "_softmax_ckc" in line:
                offenders.append("%s:%d" % (path.relative_to(ROOT), i))
    assert offenders == [], "ttnn.softmax still called directly at: %s" % ", ".join(offenders)


def test_site_softmax_off_is_ttnn_softmax_and_nothing_else(monkeypatch):
    """The byte-identity argument in one assertion: off, the wrapper forwards every argument it
    was given to `ttnn.softmax` and returns its result unchanged."""
    import tt_bio.tenstorrent as T

    seen = {}

    def fake(x, **kw):
        seen["x"], seen["kw"] = x, kw
        return "OUT"

    monkeypatch.setattr(T.ttnn, "softmax", fake)
    out = T.site_softmax("SC", dim=-1, numeric_stable=True, compute_kernel_config="CKC",
                         host_f64=False)
    assert out == "OUT"
    assert seen == {"x": "SC", "kw": {"dim": -1, "numeric_stable": True,
                                      "compute_kernel_config": "CKC"}}


def test_site_softmax_on_does_not_reach_ttnn_softmax(monkeypatch):
    import tt_bio.tenstorrent as T

    def fake(*a, **k):
        raise AssertionError("ttnn.softmax ran with the host float64 path selected")

    monkeypatch.setattr(T.ttnn, "softmax", fake)
    monkeypatch.setattr(T, "host_f64_softmax", lambda x, dim=-1: ("HOST", x, dim))
    assert T.site_softmax("SC", dim=-1, compute_kernel_config="CKC",
                          host_f64=True) == ("HOST", "SC", -1)


def test_the_census_counts_both_answers(monkeypatch):
    """A lever that fires and is inert is the failure mode this campaign keeps meeting, so the
    path publishes served and declined rather than leaving either invisible."""
    import tt_bio.tenstorrent as T

    monkeypatch.setattr(T.ttnn, "softmax", lambda x, **kw: "OUT")
    monkeypatch.setattr(T, "host_f64_softmax", lambda x, dim=-1: "HOST")
    before = dict(T.HOST_F64_SOFTMAX_STATS)
    T.site_softmax("SC", host_f64=False)
    T.site_softmax("SC", host_f64=True)
    assert T.HOST_F64_SOFTMAX_STATS["declined"] == before["declined"] + 1


def test_the_backward_reads_the_float64_forward_not_the_device_copy():
    """The whole point of the path. Taking the Jacobian on the rounded output that went back to
    the card would put a device-precision softmax straight back into the gradient."""
    src = (ROOT / "tt_bio/tenstorrent.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "host_f64_softmax")
    bw = next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == "bw")
    names = {n.id for n in ast.walk(bw) if isinstance(n, ast.Name)}
    assert "y64" in names, "the backward does not read the float64 forward output"

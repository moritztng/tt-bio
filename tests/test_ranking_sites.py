"""No site kept a private copy of the ranking rule.

`tests/test_sample_ranking.py` (of3t-rankunify) proves the RULE: with an interface every site
gets back the expression it had, without one ipTM's 0.8 goes to pLDDT. It calls the shared
function and RF3's wrapper directly, which is everything a unit test can reach.

What it cannot reach is the other two call sites. `OpenFold3._confidence` and
`worker._WorkerState._protenix_emit._score` sit inside methods that need a device and a fold, so
a site quietly rewired to a local expression would pass every test in that file. Landing the
unified rule means proving that did not happen and cannot happen later, so this file checks the
sites structurally, in both directions:

  * nothing in `tt_bio/` outside `ranking.py` computes an AF3 ranking expression of its own, and
  * every assignment at a call site that produces a score is a call INTO `tt_bio.ranking`, with
    the keywords that site is supposed to pass.

Both are negative-controlled in `perf/of3t_d10d24unify/`: reverting OpenFold3's site to
`0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100.0 * has_clash` fails both.
"""
import ast
import io
import pathlib
import re
import tokenize

SRC = pathlib.Path(__file__).resolve().parents[1] / "tt_bio"
AF3_EXPR = re.compile(r"0\.8\s*\*\s*\w+\s*\+\s*0\.2\s*\*")


def _code_only(path):
    """Comments and docstrings describe the rule all over this package, so the scan has to read
    code. Without this it fires on three prose mentions in `main.py` and `worker.py` and on
    nothing else, which is a guard whose only remedy is to reword an accurate comment."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            out.append(tok.string)
    return " ".join(out)


def test_no_site_computes_its_own_ranking_expression():
    """Boltz-2 and BoltzGen are not in `tt_bio/` scope here by accident: they keep their own
    published rule deliberately, users benchmark against it, and on a monomer it already agrees
    (`(4*pLDDT + pTM)/5 == 0.8*pLDDT + 0.2*pTM`)."""
    offenders = [str(p.relative_to(SRC.parent)) for p in sorted(SRC.rglob("*.py"))
                 if "_vendor" not in p.parts and p.name != "ranking.py"
                 and AF3_EXPR.search(_code_only(p))]
    assert offenders == [], "an AF3 ranking expression is back at: %s" % ", ".join(offenders)


def test_the_scan_finds_a_real_expression_and_ignores_one_in_prose(tmp_path):
    """Negative control, both directions. A guard that reads no code passes the test above
    forever, and one that reads comments fails it on an accurate comment. `ranking.py` itself
    is not a usable control: it builds the score in two statements
    (`interface = 0.8 * ... ; return interface + 0.2 * ptm`), so the pattern is not in it."""
    real = tmp_path / "real.py"
    real.write_text("def score(a, b):\n    return 0.8 * a + 0.2 * b\n")
    assert AF3_EXPR.search(_code_only(real)), "the scan misses a genuine expression"

    prose = tmp_path / "prose.py"
    prose.write_text('"""confidence_score is 0.8 * plddt + 0.2 * ptm."""\n'
                     "# and 0.8 * iptm + 0.2 * ptm with an interface\n"
                     "x = 1\n")
    assert not AF3_EXPR.search(_code_only(prose)), "the scan fires on a comment or docstring"
    assert AF3_EXPR.search(prose.read_text()), "...and a plain text grep would have"


def test_each_site_assigns_its_score_from_the_shared_rule():
    """The positive half. OpenFold3 passes all five terms because it computes the RASA disorder
    term and the inter-chain clash indicator; Protenix/OpenDDE pass three because they compute
    neither, which is what makes their interface branch a no-op."""
    want = {
        "openfold3_fold.py": ("ranking_score", {"iptm", "ptm", "plddt", "disorder", "has_clash"}),
        "worker.py": ("_score", {"ptm", "iptm", "plddt"}),
    }
    for fname, (target, kwargs) in want.items():
        tree = ast.parse((SRC / fname).read_text())
        produced = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == target for t in node.targets):
                produced.append(node.value)
            elif isinstance(node, ast.FunctionDef) and node.name == target:
                produced += [n.value for n in ast.walk(node)
                             if isinstance(n, ast.Return) and n.value is not None]
        assert produced, "%s: no assignment or definition of %s" % (fname, target)
        routed = [c for c in produced
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                  and c.func.attr == "ranking_score"
                  and isinstance(c.func.value, ast.Name) and c.func.value.id == "rank"]
        assert len(routed) == len(produced), (
            "%s: %d of %d values of %s do not come from rank.ranking_score"
            % (fname, len(produced) - len(routed), len(produced), target))
        for call in routed:
            assert {k.arg for k in call.keywords} == kwargs, (
                "%s: %s passes %s, expected %s"
                % (fname, target, sorted(k.arg for k in call.keywords), sorted(kwargs)))


def test_rf3_publishes_a_rounded_score_and_orders_on_an_unrounded_one():
    """D128, the defect unification surfaced, pinned so it cannot come back. `summary()` must
    return the score unrounded (worker.py orders on it) and the 4-decimal form must be applied
    only where the published JSON is written."""
    conf = (SRC / "rf3" / "confidence.py").read_text()
    assert not re.search(r'"ranking_score":\s*round\(', conf), (
        "rf3/confidence.summary() rounds the score it returns; worker.py orders on that value, "
        "so two samples differing below 1e-4 collapse and are served in sample-index order")
    worker = (SRC / "worker.py").read_text()
    assert re.search(r"ranking_score=round\(", worker), (
        "nothing applies the published 4-decimal rounding, so summary_confidences.json changed "
        "format")

"""Negative control for tests/test_ranking_sites.py, run against the real tree.

Reverts OpenFold3's scoring site to the expression it had before unification, runs the two
guards, restores the file, and runs them again. A guard that passes in both states is checking
nothing. Committed because "we checked" is not evidence; the transcript below is.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SITE = ROOT / "tt_bio" / "openfold3_fold.py"
WIRED = """        ranking_score = rank.ranking_score(
            iptm=iptm, ptm=ptm, plddt=float(plddt_atom.mean()),
            disorder=disorder, has_clash=has_clash)"""
PRIVATE = "        ranking_score = 0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100.0 * has_clash"


def guards():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ranking_sites.py",
                        "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()[-1]


original = SITE.read_text()
assert WIRED in original, "the site is not in its wired form; refusing to run"
try:
    SITE.write_text(original.replace(WIRED, PRIVATE))
    rc_broken, line_broken = guards()
finally:
    SITE.write_text(original)
rc_ok, line_ok = guards()

print("openfold3 site reverted to its own expression: rc=%d  %s" % (rc_broken, line_broken))
print("openfold3 site restored:                       rc=%d  %s" % (rc_ok, line_ok))
ok = rc_broken != 0 and rc_ok == 0
print("NEGATIVE CONTROL %s" % ("PASS - the guards decide on the site" if ok else "FAIL"))
sys.exit(0 if ok else 1)

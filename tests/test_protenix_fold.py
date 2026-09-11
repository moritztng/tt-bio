# Capstone on-device test: the full tt_bio.protenix.Protenix.fold pipeline (atom encoder
# -> diffusion atom cache -> 10-cycle trunk -> diffusion pair/single conditioning -> EDM
# sampler -> confidence head) runs end-to-end on real v2 weights at the production sampling
# schedule and produces a physical structure, an ensemble that is not scattered, and a ranked
# sample that has not drifted. Gated on the golden feats pkls + v2 ckpt.
#
# The bounds below are ABSOLUTE. They used to be `vref <= s2s * 1.4 + 1.0`: the distance to the
# reference measured against the SAMPLER'S OWN spread, which a uniformly bad sampler passes by
# widening the bar it is judged against. The only absolute check was Rg > 10, i.e. collapse.
# Each number is the observed range over four seeds on qb1 card3 (2026-08-21) with headroom;
# `scripts/protenix_fold_e2e.py` explains what this fixture can and cannot score.
import os, re, sys, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = [os.path.expanduser(p) for p in (
    "~/protenix_ife_gold.pkl", "~/protenix_trunkin_gold.pkl", "~/protenix_ref_out.pkl",
    "~/protenix_traj.pkl", "~/protenix_ckpt/protenix-v2.pt")]

pytestmark = [
    pytest.mark.device,
    pytest.mark.skipif(not all(os.path.exists(p) for p in NEED),
                       reason="needs v2 golden feats pkls + ckpt"),
]


def test_protenix_fold_end_to_end():
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "protenix_fold_e2e.py")],
                          cwd=ROOT, capture_output=True, text=True, timeout=900)
    out = proc.stdout
    # Keep stderr. Discarding it hid a TypeError in the fixture's progress callback through three
    # release passes: the fold paid for all 10 trunk cycles and 200 diffusion steps, died one line
    # before writing its result, and every log said only "fold did not finish".
    ctx = (f"rc={proc.returncode}\n--- stdout tail ---\n{out[-2000:]}"
           f"\n--- stderr tail ---\n{proc.stderr[-2000:]}")
    assert "FOLD_E2E_DONE" in out, f"fold did not finish:\n{ctx}"
    assert "finite=True" in out, f"non-finite coords:\n{ctx}"
    def num(pat):
        m = re.search(pat, out)
        assert m, f"capstone did not report /{pat}/:\n{ctx}"
        return float(m.group(1))

    rg, rg_max = num(r"Rg ([0-9.]+) A"), num(r"Rg max ([0-9.]+) A")
    spread = num(r"ensemble spread Kabsch RMSD: min [0-9.]+ max ([0-9.]+)")
    ranked = num(r"ranked sample \d+ of \d+ vs reference Kabsch RMSD: ([0-9.]+)")
    best = num(r"best sample vs reference Kabsch RMSD: ([0-9.]+)")

    # physical: neither collapsed nor blown apart (observed 13.9-16.6 A)
    assert 10.0 < rg and rg_max < 25.0, f"unphysical Rg {rg}-{rg_max} A -- conditioning bug"
    # the samples one call returned are one ensemble, not several (observed 5.38-9.66 A)
    assert spread <= 15.0, f"ensemble spread {spread} A -- the sampler returned scattered folds"
    # the sample the model RANKED FIRST is what a user gets (observed 8.00-8.48 A)
    assert ranked <= 12.0, f"ranked sample is {ranked} A from the reference prediction"
    # and ranking must not be costing more than the ensemble is wide (observed 8.00 vs 5.61-6.33)
    assert ranked - best <= spread, (
        f"rank 0 is {ranked} A out while {best} A was available in the same batch, a gap wider "
        f"than the whole ensemble ({spread} A) -- confidence is not ranking")

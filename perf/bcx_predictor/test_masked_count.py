"""masked_residue_count must see BindCraft 2's token-axis pad, not only its chain pad.

Regression for the defect bcx-mono found: the count returned 0 for the single-chain PD-L1
fold that reads 0.534 pLDDT with a 19.28% zero pair mask, because it looked only at
pad_design_chains. `predict` pads the TOTAL token axis (bindcraft/af2.py:298) and only
pads chains when target_pad_length is set.
"""
import pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bc2_state as B
from ttbio_predictor import masked_residue_count, refuse_masked_state
from bindcraft.af2 import campaign_length_bucket

s = B.campaign_settings()
bucket = campaign_length_bucket(s)
_, states, _ = B.design_state(s)
target_only = {"hPDL1": {k: v for k, v in states["hPDL1"].items() if k != "binder"}}
binder_only = {"hPDL1": {k: v for k, v in states["hPDL1"].items() if k == "binder"}}

fails = []


def check(name, got, want):
    ok = got == want
    print(("PASS " if ok else "FAIL ") + f"{name}: got {got}, want {want}")
    if not ok:
        fails.append(name)


# 115-residue target alone: predict pads the token axis 115 -> 128.
check("target alone, predict path", masked_residue_count(target_only, bucket), 128 - 115)
# 77-residue binder alone: 77 -> 96.
check("binder alone, predict path", masked_residue_count(binder_only, bucket), 96 - 77)
# The two-chain complex is 77 + 115 = 192, already a multiple of 32, so predict pads nothing.
check("two-chain complex, predict path", masked_residue_count(states, bucket), 0)
# The gradient path pads the DESIGN chain instead: 77 -> 96 with the target untouched.
check("two-chain complex, gradient path",
      masked_residue_count(states, bucket, path="sequence_gradients"), 96 - 77)
# And the guard must now refuse the fold it used to wave through.
try:
    refuse_masked_state(target_only, bucket)
    check("guard refuses the 0.534 fold", "passed", "raised")
except ValueError:
    check("guard refuses the 0.534 fold", "raised", "raised")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)

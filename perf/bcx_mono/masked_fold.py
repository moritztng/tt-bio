#!/usr/bin/env python3
"""The positive control: one natural monomer, both arms, with and without the pair mask.

`pos_control.json` read pLDDT 0.534 against BindCraft 2's own 0.950 on the 115-residue PD-L1
target folded alone, and 1.016 of BindCraft 2 on the two-chain complex. `capture.json` says why
the complex is the one that agrees: BindCraft 2 buckets the token axis to 32 and 77 + 115 = 192
is already a multiple of 32, so that fold carries an all-ones pair mask, while the single-chain
folds at 115 and 77 pad to 128 and 96 and carry a pair mask 19.3% and 35.7% zero. `splice.py`
handed the device Evoformer only `masks["msa"]`, so the pair track ran unmasked on every padded
fold.

Three arms in one process, same card, same weights:

  jax                BindCraft 2's own trunk, the reference
  pairmask_dropped   the device trunk on an all-ones pair mask -- the shipped splice, reproduced
                     by feeding ones rather than by keeping the old code, so the two device arms
                     differ by exactly this tensor
  pairmask_on        the device trunk on BindCraft 2's own pair mask

pLDDT is BindCraft 2's own metric. The target-alone fold is also scored as CA RMSD in Angstrom
against the campaign's PD-L1 structure -- the lab coordinates BindCraft 2 loaded -- aligned by
BindCraft 2's own Kabsch.
"""
import argparse, json, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor"),
          str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import bc2_state as B
import afgrad as A
import stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft.af2 import campaign_length_bucket
from bindcraft.protein import ATOM_INDEX, kabsch

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
CA = ATOM_INDEX["CA"]


class DroppedPairMask(EvoformerOnDevice):
    """The shipped splice: the pair mask replaced by ones on its way to the card.

    `af2_pair_masks` returns `(None, None)` on an all-ones mask, so this arm runs the exact
    arithmetic the pair track ran before the mask was plumbed, in the same process as the arm
    that fixes it.
    """

    def _pair_mask(self, pm):
        return super()._pair_mask(pm.new_ones(pm.shape))


def split_plddt(pred, state="hPDL1"):
    sp = pred[state]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    out, i = {}, 0
    for name in sorted(sp.protein_complex):
        n = len(sp.protein_complex[name])
        out[name] = round(float(plddt[i:i + n].mean()), 4)
        i += n
    out["_all"] = round(float(plddt.mean()), 4)
    out["_ptm"] = round(float(sp.metrics["ptm"]), 4)
    return out


def ca_rmsd(pred, truth, mask):
    """Kabsch-aligned CA RMSD in Angstrom, BindCraft 2's own alignment."""
    p = np.asarray(pred, np.float64)[mask]
    t = np.asarray(truth, np.float64)[mask]
    rot, pc, tc = kabsch(p, t, np.ones(p.shape[0]))
    aligned = np.asarray((p - pc) @ np.asarray(rot)) + np.asarray(tc)
    return float(np.sqrt(((aligned - t) ** 2).sum(-1).mean()))


ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="*", default=["jax", "pairmask_dropped", "pairmask_on"])
ap.add_argument("--cases", nargs="*", default=["target_alone", "complex", "binder_alone"])
ap.add_argument("--card", type=int, default=0)
ap.add_argument("--out", default=str(HERE / "masked_fold.json"))
a = ap.parse_args()

s = B.campaign_settings()
bucket = campaign_length_bucket(s)
_, states, _ = B.design_state(s)
CASES = {
    "complex": states,
    "target_alone": {"hPDL1": {k: v for k, v in states["hPDL1"].items() if k != "binder"}},
    "binder_alone": {"hPDL1": {k: v for k, v in states["hPDL1"].items() if k == "binder"}},
}
TRUTH = states["hPDL1"]["target_hPDL1"]
truth_ca = np.asarray(TRUTH.atoms)[:, CA]
truth_ok = np.asarray(TRUTH.atom_mask)[:, CA].astype(bool)


def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                     max_cache_size=4, trunk="jax")


def score(pred, case):
    got = {"plddt": split_plddt(pred)}
    if case in ("target_alone", "complex"):
        p = np.asarray(pred["hPDL1"].protein_complex["target_hPDL1"].atoms)[:, CA]
        n = min(p.shape[0], truth_ca.shape[0])
        got["target_ca_rmsd_A"] = round(ca_rmsd(p[:n], truth_ca[:n], truth_ok[:n]), 3)
    return got


out = {"bucket": bucket, "arms": {}}
spans = []
clock = S.Clock()

if "jax" in a.arms:
    res = {}
    for c in a.cases:
        t0 = time.time(); pred = model().predict(CASES[c]); spans.append((t0, time.time()))
        res[c] = score(pred, c)
        print("jax", c, json.dumps(res[c]), flush=True)
    out["arms"]["jax"] = res

device_arms = [x for x in a.arms if x != "jax"]
if device_arms:
    dm, _ref = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm)
    lv = S.Levers(); lv.arm("stack")
    for arm in device_arms:
        cls = EvoformerOnDevice if arm == "pairmask_on" else DroppedPairMask
        evo = cls(dev, k_evo=48)
        res = {}
        with evoformer_on_device(evo):
            for c in a.cases:
                t0 = time.time(); pred = model().predict(CASES[c]); spans.append((t0, time.time()))
                res[c] = score(pred, c)
                print(arm, c, json.dumps(res[c]), flush=True)
        res["_primal_calls"] = evo.calls["primal"]
        out["arms"][arm] = res

clock.stop()
out["aiclk"] = clock.window(spans)
out["stamp"] = A.stamp(a.card)
pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))

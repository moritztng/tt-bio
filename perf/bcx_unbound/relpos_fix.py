"""The two instruments hand the monomer model DIFFERENT residue numbering, and that is the
whole harden-to-mutate step.

`_predict_complex` renumbers a multi-chain complex for a monomer model before it builds the
features (af2.py:305-306 -> `monomer_chain_break_indices`, MONOMER_CHAIN_GAP 49). The
gradient path builds asym_id, entity_id and seq_mask the same way and never calls it
(af2.py:337-345). model_1_ptm is a monomer model (af2.py:420) and the design complex has two
chains, so the branch is live for every step of this campaign.

It matters because the monomer relpos consumes `offset` alone, clipped to +-32
(modules.py:1469-1484); asym_id never reaches it. Renumbered, every cross-chain pair sits at
|offset| >= 50 and clips to the extreme bin. Numbered straight through, the binder ends at
its own length and the target restarts at 18, so the junction step is NEGATIVE and a wide
band of binder x target pairs lands INSIDE the +-32 window -- the monomer model is told those
residues are sequence neighbours.

Four legs. `fix` forces every cross-chain offset outside the clip window, which is
bin-identical to what the renumbering already does, so:

  grad  unpatched -> the gradient stages' own reading
  grad  patched   -> if it moves onto `predict`, the numbering is the carrier
  predict unpatched -> the mutate stage's reading
  predict patched   -> MUST equal it. That is the control proving the patch only does what
                       the renumbering already did, rather than changing the prediction.

Then the three-sequence probe through `predict` at full precision, and the same sequence read
again at the end of a process that has run two taped gradient calls (CARRIED; `--fresh` runs
only that one leg, in a process that has opened the card and nothing else).
"""
import json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor"),
          str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)

import numpy as np, jax, jax.numpy as jnp
import bc2_state as B
import afgrad as A, stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft import af2 as BAF2
from bindcraft.target_schedule import losses_for_active_states
from bindcraft.af.alphafold.common import residue_constants

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
SP = dict(softmax_weight=1.0, one_hot_weight=1.0, temperature=0.01, logit_scale=2.0)
CLIP = 32                       # af/alphafold/model/config.py:226 max_relative_feature
FRESH = "--fresh" in sys.argv
PROBES = "--probes" in sys.argv
TAG = "fresh" if FRESH else ("probes" if PROBES else "full")

# ------------------------------------------------------------------ the patch, off by default
FIX = {"on": False}
_real_offsets = BAF2.cyclic_sequence_offsets


def _patched_offsets(residue_index, asym_id, flags, seq_mask, offset_mode="direction"):
    offsets = _real_offsets(residue_index, asym_id, flags, seq_mask, offset_mode)
    if not FIX["on"]:
        return offsets
    cross = asym_id[:, None] != asym_id[None, :]
    outside = jnp.where(asym_id[:, None] < asym_id[None, :], -(CLIP + 1), CLIP + 1)
    return jnp.where(cross, outside, offsets).astype(offsets.dtype)


BAF2.cyclic_sequence_offsets = _patched_offsets

s = B.campaign_settings()
_ds, states, losses = B.design_state(s)
active_losses = losses_for_active_states(losses, states)
binder = states["hPDL1"]["binder"]
seq = np.asarray(binder.sequence)
n_res, n_aa = seq.shape
base_idx = seq.argmax(-1)


def one_hot(idx):
    out = np.zeros((n_res, n_aa), dtype=seq.dtype)
    out[np.arange(n_res), idx] = 1.0
    return jnp.asarray(out)


# ------------------------------------------------- the mechanism, counted from the inputs alone
def offset_census():
    from bindcraft.prediction import concatenate_chain_arrays, residue_chain_ids
    pc = states["hPDL1"]
    names = tuple(sorted(pc))
    out = {}
    for tag, complex_, renumber in (
            ("gradient_path", BAF2.pad_design_chains({"hPDL1": pc}, 32, 0)["hPDL1"], False),
            ("predict_path", pc, True)):
        lengths = tuple(len(complex_[n]) for n in names)
        ri = np.asarray(concatenate_chain_arrays(names, complex_, "residue_index")["residue_index"])
        if renumber:
            ri = np.asarray(BAF2.monomer_chain_break_indices(lengths, jnp.asarray(ri)))
        asym = np.asarray(residue_chain_ids(lengths))
        off = ri[:, None] - ri[None, :]
        cross = asym[:, None] != asym[None, :]
        inside = cross & (np.abs(off) <= CLIP)
        j = int(np.cumsum(lengths[:-1])[0])
        out[tag] = {"n": int(ri.shape[0]), "chain_lengths": list(lengths),
                    "junction_step": int(ri[j]) - int(ri[j - 1]),
                    "cross_chain_pairs": int(cross.sum()),
                    "cross_pairs_inside_clip": int(inside.sum()),
                    "fraction_inside_clip": round(float(inside.sum() / max(cross.sum(), 1)), 4),
                    "cross_pairs_at_offset_zero": int((cross & (off == 0)).sum())}
    return out


def model():
    # the mutate stage's own settings: design model, design_recycles 1, dropout False
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=32,
                                     max_cache_size=6, dropout=False, trunk="jax")


def read(pred):
    sp = pred["hPDL1"]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    ptm, iptm = float(np.asarray(sp.metrics["ptm"])), float(np.asarray(sp.metrics["iptm"]))
    return {"n": int(plddt.shape[0]), "ptm": round(ptm, 6), "iptm": round(iptm, 6),
            "ptm_csv": round(ptm, 2), "iptm_csv": round(iptm, 2),
            "binder_plddt": round(float(plddt[:n_res].mean()), 4),
            "plddt": round(float(plddt.mean()), 4)}


out = {"why": "the gradient path never renumbers the complex for the monomer model; predict does",
       "source": {"predict_renumbers": "bindcraft/af2.py:305-306",
                  "gradient_path_does_not": "bindcraft/af2.py:337-345",
                  "relpos_uses_offset_only": "af/alphafold/model/config.py:226 clip 32, "
                                             "af/alphafold/model/modules.py:1469-1484",
                  "monomer_chain_gap": int(BAF2.MONOMER_CHAIN_GAP)},
       "sequence_parameters": SP, "dropout": False, "num_recycle": 1, "arm": "device",
       "binder_len": int(n_res), "census": offset_census(), "result": {}}
print(json.dumps(out["census"], indent=1), flush=True)

variants = {"v0_base": (jnp.asarray(seq), None)}
for k, pos in enumerate(np.linspace(5, n_res - 6, 3).astype(int)):
    idx = base_idx.copy()
    idx[pos] = (int(base_idx[pos]) + 5 + k) % n_aa
    variants[f"v{k + 1}_pos{pos}"] = (one_hot(idx),
                                      f"{residue_constants.restypes[int(base_idx[pos])]}{pos}"
                                      f"{residue_constants.restypes[int(idx[pos])]}")
variants["v4_polyala"] = (one_hot(np.full(n_res, residue_constants.restypes.index("A"))), "poly-A")
out["sequences"] = {k: v[1] for k, v in variants.items()}


def state_for(sq):
    return {"hPDL1": {**states["hPDL1"], "binder": binder.replace(sequence=sq)}}


def leg(tag, fn):
    t0 = time.time()
    res = fn()
    res["s"] = round(time.time() - t0, 1)
    out["result"][tag] = res
    print(tag, json.dumps(res), flush=True)
    (HERE / f"relpos_fix_{TAG}.json").write_text(json.dumps(out, indent=1, default=str))
    return res


lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); w0 = time.time()
with evoformer_on_device(evo):
    m = model()
    if FRESH:
        # CARRIED, fresh half: the first model call this process makes is the read itself.
        leg("predict_fresh_process", lambda: read(m.predict(states, **SP)))
    elif PROBES:
        for name, (sq, _label) in variants.items():
            leg(f"probe_{name}", lambda sq=sq: read(m.predict(state_for(sq), **SP)))
    else:
        FIX["on"] = False
        leg("grad_unpatched", lambda: read(m.sequence_gradients(states, active_losses, **SP)[0]))
        leg("predict_unpatched", lambda: read(m.predict(states, **SP)))
        FIX["on"] = True
        leg("grad_patched", lambda: read(m.sequence_gradients(states, active_losses, **SP)[0]))
        leg("predict_patched", lambda: read(m.predict(states, **SP)))
        FIX["on"] = False
        for name, (sq, _label) in variants.items():
            leg(f"probe_{name}", lambda sq=sq: read(m.predict(state_for(sq), **SP)))
        # CARRIED, used half: the same call as predict_unpatched, after two taped gradient
        # calls and five untaped predicts in this process.
        leg("predict_used_process", lambda: read(m.predict(states, **SP)))
w1 = time.time(); clock.stop()
out["aiclk_during"] = clock.window([(w0, w1)])
out["device_calls"] = dict(evo.calls)
out["sysfs_node"] = list(S.sysfs_node())
out["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]))

r = out["result"]
if PROBES:
    out["verdict"] = {
        "probe_distinct_ptm_full": sorted({r[f"probe_{k}"]["ptm"] for k in variants}),
        "probe_distinct_ptm_csv": sorted({r[f"probe_{k}"]["ptm_csv"] for k in variants}),
        "probe_distinct_iptm_full": sorted({r[f"probe_{k}"]["iptm"] for k in variants}),
        "probe_shift_from_base": {k: {"d_ptm": round(r[f"probe_{k}"]["ptm"] - r["probe_v0_base"]["ptm"], 6),
                                      "d_iptm": round(r[f"probe_{k}"]["iptm"] - r["probe_v0_base"]["iptm"], 6),
                                      "moves_ptm_at_csv_resolution": r[f"probe_{k}"]["ptm_csv"] != r["probe_v0_base"]["ptm_csv"]}
                                  for k in variants if k != "v0_base"}}
    print(json.dumps(out["verdict"], indent=1), flush=True)
if not (FRESH or PROBES):
    out["verdict"] = {
        "instrument_gap_unpatched": {k: round(r["grad_unpatched"][k] - r["predict_unpatched"][k], 4)
                                     for k in ("iptm", "ptm", "plddt", "binder_plddt")},
        "instrument_gap_patched": {k: round(r["grad_patched"][k] - r["predict_patched"][k], 4)
                                   for k in ("iptm", "ptm", "plddt", "binder_plddt")},
        "predict_patch_is_a_noop": {k: round(r["predict_patched"][k] - r["predict_unpatched"][k], 6)
                                    for k in ("iptm", "ptm", "plddt", "binder_plddt")},
        "probe_distinct_ptm_full": sorted({r[f"probe_{k}"]["ptm"] for k in variants}),
        "probe_distinct_ptm_csv": sorted({r[f"probe_{k}"]["ptm_csv"] for k in variants}),
        "probe_distinct_iptm_full": sorted({r[f"probe_{k}"]["iptm"] for k in variants}),
        "carried_used_minus_first": {k: round(r["predict_used_process"][k] - r["predict_unpatched"][k], 6)
                                     for k in ("iptm", "ptm", "plddt", "binder_plddt")},
    }
    print(json.dumps(out["verdict"], indent=1), flush=True)
print("AICLK", json.dumps(out["aiclk_during"], default=str), "calls", out["device_calls"], flush=True)
(HERE / f"relpos_fix_{TAG}.json").write_text(json.dumps(out, indent=1, default=str))

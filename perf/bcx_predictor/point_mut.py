"""Does a SINGLE point mutation move pTM at the resolution BindCraft 2 records?

post_seed100's mutate stage logged pTM 0.59 on all fifteen rounds and this row read that as
a metric that had stopped tracking its input. Two things have to be true for that reading to
survive, and this tests the second.

The first is settled by source. `run_sequence_mutation_stage` (trajectory.py:158-166) scores
the CANDIDATE -- `predictions = design_model.predict(candidate_states)` -- and hands that same
object to the recorder, so the log carries the proposal and not the incumbent. An earlier note
in this row's doc guessed the opposite; it was wrong and is withdrawn.

The second is resolution. Every value in `*_losses.csv` is written to two decimals, so the
log's resolution is 0.005, and each proposal differs from the incumbent by ONE residue in 71.
If a single point mutation does not move pTM by 0.005 in BindCraft 2's own predictor, then
fifteen identical 0.59s are what the file would show even with everything working.

JAX only. No device: the card is busy with the bucket-1 control trajectory, and
`instrument.json` already puts the device within 0.0035 of this arm on both instruments, so
the device cannot be the variable here. Threads are capped so the trajectory keeps its cores.

Caveat stated up front: the mutate stage mutates an OPTIMISED binder and no run on disk saved
one, so this mutates the seed-0 initial draw. Its pTM is 0.5696, close to the 0.59 being
explained, but sensitivity to a point mutation may differ on an optimised sequence.
"""
import json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)

import numpy as np, jax, jax.numpy as jnp
import bc2_state as B
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft.af.alphafold.common import residue_constants

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
N_MUT = 4

s = B.campaign_settings()
_ds, states, _ = B.design_state(s)
binder = states["hPDL1"]["binder"]
seq = np.asarray(binder.sequence)
n_res, n_aa = seq.shape
base = seq.argmax(-1)


def one_hot(idx):
    out = np.zeros((n_res, n_aa), dtype=seq.dtype)
    out[np.arange(n_res), idx] = 1.0
    return jnp.asarray(out)


# One residue changed per variant, spread across the binder, each to a different amino acid.
variants = {"v0_unmutated": (jnp.asarray(seq), None)}
for k, pos in enumerate(np.linspace(5, n_res - 6, N_MUT).astype(int)):
    idx = base.copy()
    idx[pos] = (int(base[pos]) + 5 + k) % n_aa
    variants[f"v{k + 1}_pos{pos}"] = (
        one_hot(idx),
        f"{residue_constants.restypes[int(base[pos])]}{pos}"
        f"{residue_constants.restypes[int(idx[pos])]}")


def model():
    # The mutate stage's own settings: design model, design_recycles 1, dropout False.
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=32,
                                     max_cache_size=4, dropout=False, trunk="jax")


def summarise(pred):
    sp = pred["hPDL1"]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    ptm = float(np.asarray(sp.metrics["ptm"]))
    iptm = float(np.asarray(sp.metrics["iptm"]))
    return {"ptm_full": round(ptm, 6), "iptm_full": round(iptm, 6),
            "ptm_csv": round(ptm, 2), "iptm_csv": round(iptm, 2),
            "binder_plddt": round(float(plddt[:n_res].mean()), 4),
            "plddt": round(float(plddt.mean()), 4)}


out = {"why": "post_seed100 logged pTM 0.59 on all 15 mutate rounds; the log is written to 2 decimals",
       "source_fact": "trajectory.py:161-166 scores the CANDIDATE and records that object, "
                      "so the log carries the proposal, not the incumbent",
       "csv_resolution": 0.005, "binder_len": n_res, "arm": "jax",
       "mutations": {k: v[1] for k, v in variants.items()}, "result": {}}
print(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1), flush=True)

m = model()
for name, (sq, label) in variants.items():
    st = {"hPDL1": {**states["hPDL1"], "binder": binder.replace(sequence=sq)}}
    t0 = time.time()
    out["result"][name] = summarise(m.predict(st))
    out["result"][name]["s"] = round(time.time() - t0, 1)
    print(name, label, json.dumps(out["result"][name]), flush=True)
    (HERE / "point_mut.json").write_text(json.dumps(out, indent=1, default=str))

mutated = [k for k in out["result"] if k != "v0_unmutated"]
b = out["result"]["v0_unmutated"]
out["shift_from_unmutated"] = {
    k: {"d_ptm": round(out["result"][k]["ptm_full"] - b["ptm_full"], 6),
        "d_iptm": round(out["result"][k]["iptm_full"] - b["iptm_full"], 6),
        "moves_ptm_at_csv_resolution": out["result"][k]["ptm_csv"] != b["ptm_csv"],
        "moves_iptm_at_csv_resolution": out["result"][k]["iptm_csv"] != b["iptm_csv"]}
    for k in mutated}
out["verdict"] = {
    "ptm_csv_values": sorted({out["result"][k]["ptm_csv"] for k in out["result"]}),
    "iptm_csv_values": sorted({out["result"][k]["iptm_csv"] for k in out["result"]}),
    "max_abs_d_ptm": round(max(abs(v["d_ptm"]) for v in out["shift_from_unmutated"].values()), 6),
    "max_abs_d_iptm": round(max(abs(v["d_iptm"]) for v in out["shift_from_unmutated"].values()), 6)}
out["post_seed100_mutate"] = {"ptm": "0.59 x15, 1 distinct value",
                              "iptm": "0.10/0.11, 2 distinct values",
                              "anneal_ptm_for_comparison": "9 distinct over 45 rounds, longest constant run 5",
                              "harden_ptm_for_comparison": "2 distinct over 5 rounds, longest constant run 4"}
(HERE / "point_mut.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out["verdict"], indent=1), flush=True)
print(json.dumps(out["shift_from_unmutated"], indent=1), flush=True)

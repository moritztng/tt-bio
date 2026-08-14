#!/usr/bin/env python3
"""Name the 2.576 s inside `top:expand_and_refine` that nothing has attributed.

perf/odde4x/decomp_512.json measures top:expand_and_refine at 3.687 s and attributes only
expander:call 1.065 + expander:pair_project_full 0.046. The rest is the ledger's "trunk glue /
expander seam" row, which the state doc says is still mostly unnamed.

Reading opendde.py:417-445, the function has four phases and only the first is timed:
    expander   self.expander(ifd, ...)          -> already timed as expander:call
    prewarm    8 x trimul.prewarm(Ns, 1)        -> builds the 96-tensor _gp_cache. Setup, and
                                                  "numerically inert" per the code comment, so if
                                                  it costs anything on a WARM fold it is a pure
                                                  deletable if the cache can outlive the fold
    reshape    4 x ttnn.reshape                 -> expected free
    refiner    self.refiner(s3, z4, bias)       -> 4 blocks at the structural axis. NOT timed
                                                  anywhere. Real compute, not glue, if it is big

That split decides whether the row is a deletable setup cost or simply more compute nobody had
labelled. Device legs are synchronised so the numbers are device time, not enqueues.

Faithfulness: the replica must not change the computation. The run asserts plDDT 0.75411 and the
full 64-hex CIF 357c67003bb738ac...7001d92b, the reference for this card.
"""
import hashlib, json, sys, time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/opendde-beat-b200")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "gpu_vs_tt"))

import tt_baseline as TB
import ttnn
import tt_bio.tenstorrent as T
from tt_bio.opendde import OpenDDE

REC = []
REF_PLDDT = 0.75411
REF_CIF = "357c67003bb738ac"
_orig = OpenDDE.expand_and_refine


def phased(self, ifd, s_inputs_res, s_res, z_res, *, extra_attn_bias=True,
           return_attn_bias=False):
    dev = T.get_device()
    ph = {}

    def mark(k, t0):
        ttnn.synchronize_device(dev)
        ph[k] = round(time.perf_counter() - t0, 4)

    t = time.perf_counter()
    s_inputs_st, s_st, z_st, attn_bias = self.expander(ifd, s_inputs_res, s_res, z_res)
    mark("expander", t)
    Ns = s_st.shape[0]

    t = time.perf_counter()
    for blk in self.refiner.blocks:
        blk.triangle_multiplication_start.prewarm(Ns, 1)
        blk.triangle_multiplication_end.prewarm(Ns, 1)
    mark("prewarm", t)

    t = time.perf_counter()
    z4 = ttnn.reshape(z_st, (1, Ns, Ns, self.expander.c_z))
    s3 = ttnn.reshape(s_st, (1, Ns, self.expander.c_s))
    bias = None
    if extra_attn_bias:
        bias = ttnn.reshape(attn_bias, (1, 1, Ns, Ns))
    mark("reshape", t)

    t = time.perf_counter()
    s_ref, z_ref = self.refiner(s3, z4, extra_attn_bias=bias)
    mark("refiner", t)

    t = time.perf_counter()
    result = (s_inputs_st, ttnn.reshape(s_ref, (Ns, self.expander.c_s)), z_ref)
    mark("tail", t)

    ph["total"] = round(sum(v for k, v in ph.items()), 4)
    ph["Ns"] = int(Ns)
    REC.append(ph)
    if return_attn_bias:
        return (*result, attn_bias)
    return result


OpenDDE.expand_and_refine = phased

FIX = WT / "perf" / "size512" / "fixtures"
one_fold, meta, state = TB.build_fold(
    "opendde", WT / ".msa_om512_512", FIX / "cdk2x2_512.yaml", FIX / "cdk2x2_512.a3m")

folds = []
for i in range(3):
    t, m = one_fold()
    cifs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(meta["struct_dir"]).glob("*.cif"))}
    folds.append({"fold": i, "kind": "cold" if i == 0 else "warm", "fold_s": round(t, 3),
                  "plddt": m.get("plddt"), "cif_sha256_full": cifs,
                  "phases": REC[-1] if REC else None})
    print(f"fold {i} {folds[-1]['kind']:4} {t:8.3f}s plddt {m.get('plddt')} {REC[-1]}", flush=True)

bad = [f for f in folds if abs((f["plddt"] or 0) - REF_PLDDT) > 1e-9
       or not all(v.startswith(REF_CIF) for v in f["cif_sha256_full"].values())]
warm = [f for f in folds if f["kind"] == "warm"]
keys = ["expander", "prewarm", "reshape", "refiner", "tail", "total"]
avg = {k: round(sum(f["phases"][k] for f in warm) / len(warm), 4) for k in keys}
cold = folds[0]["phases"]

summary = {
    "host": "tt-quietbox2", "model": "opendde", "n_tokens": 512, "Ns": cold.get("Ns"),
    "replica_faithful": not bad,
    "ref": {"plddt": REF_PLDDT, "cif_sha256_prefix": REF_CIF},
    "folds": folds,
    "warm_phase_avg_s": avg,
    "cold_phases_s": cold,
    "inherited_decomp": {"top:expand_and_refine": 3.687, "expander:call": 1.065,
                         "expander:pair_project_full": 0.046,
                         "unattributed": round(3.687 - 1.065 - 0.046, 4)},
}
p = WT / "perf" / "oddeb200" / "screen_expand.json"
p.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: summary[k] for k in
                  ("replica_faithful", "Ns", "warm_phase_avg_s", "cold_phases_s",
                   "inherited_decomp")}, indent=1))
if bad:
    print("REPLICA DRIFTED -- phase numbers are NOT valid", file=sys.stderr)
T.cleanup()

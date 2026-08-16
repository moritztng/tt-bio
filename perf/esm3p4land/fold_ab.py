#!/usr/bin/env python3
"""Fold-level A/B for the two ported levers, on any model that reaches them.

Arms, set explicitly on every run so no arm inherits the previous one's state:

  base    the port's default: the E6 gated move re-keyed, L1-resident fc1 OFF
  l2      + `esmc.set_pair_ffn_l1_fc1(True)`

Run the same command on `origin/main` (with `--arms base`) for the gated-move leg. `base` there
and `base` here differ only in the re-key, so the CIF hashes must match across the two processes
and the wall-clock difference is what the re-key is worth.

Arms repeat round-robin so host drift hits each equally, and `base`'s own spread IS the A/A floor.
"""
import argparse, hashlib, json, os, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

# `ship` names no lever, so it runs the shipped defaults snapshotted at process start (see
# SHIPPED below). That is the arm the page publishes, and the only one that catches a default
# that did not land.
ARMS = {"base": False, "l2": True, "ship": None, "f1": None, "nof1": None,
        "B": None, "D": None, "BD": None, "off": None, "E": None, "noE": None,
        "Cin": None, "F": None, "CinF": None, "noCinF": None}

# B, the LM shim -> LM encoder handoff (esmfold2 only). Every arm sets the gate on every fold:
# a lever arm to what it names, any other arm back to the snapshotted shipped default. Neither
# `.get(arm, False)` (which forces the gate off and makes `ship` blind to a default that did not
# land) nor leaving the gate alone (which hands `ship` whatever the previous arm mutated) works.
# `off` is the pre-lever baseline. It is needed because both levers now ship ON, which makes
# `ship` and `BD` the same arm -- without an explicit both-off arm there is nothing to measure the
# delta against on a tree where the default has already landed.
DEVPAIR = {"B": True, "BD": True, "D": False, "off": False}

# D: half the trimul in-projection's output drain on the other NOC (tt_bio/mm_dualnoc.py).
DUALNOC = {"D": True, "BD": True, "B": False, "off": False}

# E: the L1 destination on the pair FFN's block layer_norm (esmc.set_pair_ffn_l1_ln). E ships ON,
# so `ship` already contains it and the arm that isolates it is `noE`: ship - noE is E on the
# current tree, with B and D where they ship. `off` stays the all-levers-off baseline.
L1LN = {"E": True, "noE": False, "off": False}

# C-in (esmc.set_pair_ffn_l1_slice): the row block sliced lazily into L1 instead of chunked into
# DRAM. F (esmc.set_pair_ffn_fused_residual): the pair transition's residual add folded into the
# block so fc2 writes L1. Both ship ON, so the isolating arm is `noCinF` and ship - noCinF is the
# pair on the current tree. Same rule as L1LN: every arm writes BOTH gates every round, because
# they compose -- an arm that names C-in and leaves F wherever the previous arm left it is not
# the C-in arm.
L1SLICE = {"Cin": True, "CinF": True, "F": False, "noCinF": False, "off": False}
FUSEDRES = {"F": True, "CinF": True, "Cin": False, "noCinF": False, "off": False}


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="base,l2")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]

    import tt_bio.tenstorrent as T
    import tt_bio.esmc as EC
    import tt_bio.reblock_permute as RP
    import tt_bio.esmfold2_runtime as RT
    import tt_bio.mm_dualnoc as DN
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "git_head": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
           "has_l1_fc1_gate": hasattr(EC, "set_pair_ffn_l1_fc1"),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "arms": arms, "rounds": a.rounds, "runs": []}

    tgt = a.fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = a.fixdir / ("cdk2x2_%d.a3m" % a.size)
    one_fold, meta, state = B.build_fold(a.model, ROOT / (".msa_ab512_%d" % a.size), tgt, a3m)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    struct_dir = Path(meta["struct_dir"])

    # The shipped defaults, read ONCE before any arm has mutated them. `set_*` writes a module
    # global, so an arm that only "leaves the gate alone" actually inherits whatever the previous
    # arm left there -- the BD arm turns both levers on and the next `ship` then measures BD.
    # Snapshot here and restore below, so `ship` means the module default and nothing else.
    SHIPPED = {"l1_fc1": EC._PAIR_FFN_L1_FC1, "lm_handoff": RT._DEVICE_LM_HANDOFF,
               "dual_noc": DN._ENABLED,
               "l1_ln": getattr(EC, "_PAIR_FFN_L1_LN", None),
               "l1_slice": getattr(EC, "_PAIR_FFN_L1_SLICE", None),
               "fused_resid": getattr(EC, "_PAIR_FFN_FUSED_RESIDUAL", None)}
    res["shipped_defaults"] = dict(SHIPPED)

    def run(tag, arm):
        want = ARMS[arm]
        if want is None:
            want = SHIPPED["l1_fc1"]
        if want != EC._PAIR_FFN_L1_FC1:
            if not hasattr(EC, "set_pair_ffn_l1_fc1"):
                raise SystemExit("arm %s needs set_pair_ffn_l1_fc1, absent here" % arm)
            EC.set_pair_ffn_l1_fc1(want)
        RT.set_device_lm_handoff(DEVPAIR.get(arm, SHIPPED["lm_handoff"]))
        DN.set_enabled(DUALNOC.get(arm, SHIPPED["dual_noc"]))
        if SHIPPED["l1_ln"] is not None:
            EC.set_pair_ffn_l1_ln(L1LN.get(arm, SHIPPED["l1_ln"]))
            EC.L1_LN_STATS[0] = EC.L1_LN_STATS[1] = 0
        elif arm in L1LN:
            raise SystemExit("arm %s needs set_pair_ffn_l1_ln, absent here" % arm)
        if SHIPPED["l1_slice"] is not None:
            EC.set_pair_ffn_l1_slice(L1SLICE.get(arm, SHIPPED["l1_slice"]))
            EC.set_pair_ffn_fused_residual(FUSEDRES.get(arm, SHIPPED["fused_resid"]))
            EC.L1_SLICE_STATS[0] = EC.L1_SLICE_STATS[1] = 0
            EC.FUSED_RESID_STATS[0] = EC.FUSED_RESID_STATS[1] = 0
        elif arm in L1SLICE:
            raise SystemExit("arm %s needs set_pair_ffn_l1_slice, absent here" % arm)
        DN.STATS[0] = DN.STATS[1] = 0
        DN.REJECTS.clear()
        RT.LM_HANDOFF_STATS[0] = RT.LM_HANDOFF_STATS[1] = 0
        RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
        EC.L1_FC1_STATS[0] = EC.L1_FC1_STATS[1] = 0
        RP.STATS[0] = RP.STATS[1] = 0
        RP.STATS_BACK[0] = RP.STATS_BACK[1] = 0
        f1 = {"f1": True, "nof1": False}.get(arm)
        if f1 is not None:
            T._TRIMUL_TAIL_F1 = f1
            import tt_bio.trimul_tail as F1M
            F1M.STATS[0] = F1M.STATS[1] = 0
            F1M.REJECTS.clear()
        fold_s, m = one_fold()
        row = {"tag": tag, "arm": arm, "l1_fc1": want,
               "fold_s": round(fold_s, 3), "plddt": m.get("plddt"), "cif": sha_dir(struct_dir),
               "e6_served": RP.STATS_GATED[0], "e6_declined": RP.STATS_GATED[1],
               "l1_fc1_stats": list(EC.L1_FC1_STATS),
               "l1_ln": (SHIPPED["l1_ln"] is not None) and EC._PAIR_FFN_L1_LN,
               "l1_ln_stats": list(getattr(EC, "L1_LN_STATS", [])),
               "l1_slice": (SHIPPED["l1_slice"] is not None) and EC._PAIR_FFN_L1_SLICE,
               "l1_slice_stats": list(getattr(EC, "L1_SLICE_STATS", [])),
               "fused_resid": (SHIPPED["fused_resid"] is not None) and EC._PAIR_FFN_FUSED_RESIDUAL,
               "fused_resid_stats": list(getattr(EC, "FUSED_RESID_STATS", [])),
               "lm_handoff": list(RT.LM_HANDOFF_STATS),
               "fwd_move": list(RP.STATS), "back_move": list(RP.STATS_BACK),
               "l1_out_refused": len(T._L1_OUT_REFUSED),
               "dual_noc": list(DN.STATS),
               "dual_noc_rejects": {str(k): v for k, v in DN.REJECTS.items()},
               "trimul_tail_f1": (arm in ("f1", "nof1")) and list(F1M.STATS) or None,
               "loadavg": open("/proc/loadavg").read().split()[0]}
        res["runs"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
        print("  %-14s %8.3fs plddt=%s e6=%d/%d l1fc1=%d/%d lmh=%d/%d dn=%d/%d ln=%s "
              "sl=%s fr=%s l1refused=%d cif=%s load=%s"
              % (tag, fold_s, m.get("plddt"), RP.STATS_GATED[0], RP.STATS_GATED[1],
                 EC.L1_FC1_STATS[0], EC.L1_FC1_STATS[1],
                 RT.LM_HANDOFF_STATS[0], RT.LM_HANDOFF_STATS[1],
                 DN.STATS[0], DN.STATS[1],
                 "%d/%d" % tuple(getattr(EC, "L1_LN_STATS", [-1, -1])),
                 "%d/%d" % tuple(getattr(EC, "L1_SLICE_STATS", [-1, -1])),
                 "%d/%d" % tuple(getattr(EC, "FUSED_RESID_STATS", [-1, -1])),
                 row["l1_out_refused"],
                 list(row["cif"].values())[0] if row["cif"] else "-", row["loadavg"]),
              flush=True)
        return fold_s

    print("=== %s %d aa rec=%s steps=%s arms=%s rounds=%d head=%s ==="
          % (a.model, a.size, B.RECYCLING_STEPS, B.SAMPLING_STEPS, arms, a.rounds,
             res["git_head"]), flush=True)
    for arm in arms:
        run("warm:%s" % arm, arm)
    for r in range(a.rounds):
        for arm in arms:
            run("r%d:%s" % (r, arm), arm)

    by = {}
    for row in res["runs"]:
        if row["tag"].startswith("warm:"):
            continue
        by.setdefault(row["arm"], []).append(row["fold_s"])
    summary = {}
    for arm, v in by.items():
        summary[arm] = {"n": len(v), "median": round(st.median(v), 3), "min": min(v),
                        "max": max(v), "spread": round(max(v) - min(v), 3)}
    ref = "base" if "base" in summary else ("ship" if "ship" in summary else None)
    if ref:
        b = summary[ref]["median"]
        for arm, s in summary.items():
            s["speedup_vs_%s" % ref] = round(b / s["median"], 4)
            s["delta_s"] = round(b - s["median"], 3)
        summary["A/A_noise_floor_s"] = summary[ref]["spread"]
    res["summary"] = summary
    cifs = {row["arm"]: tuple(sorted(row["cif"].items())) for row in res["runs"]}
    res["cif_identical_across_arms"] = len(set(cifs.values())) == 1
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print("CIF identical across all arms: %s" % res["cif_identical_across_arms"], flush=True)


if __name__ == "__main__":
    main()

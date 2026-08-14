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

ARMS = {"base": False, "l2": True}


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

    def run(tag, arm):
        want = ARMS[arm]
        if hasattr(EC, "set_pair_ffn_l1_fc1"):
            EC.set_pair_ffn_l1_fc1(want)
        elif want:
            raise SystemExit("arm %s needs set_pair_ffn_l1_fc1, absent here" % arm)
        RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
        EC.L1_FC1_STATS[0] = EC.L1_FC1_STATS[1] = 0
        RP.STATS[0] = RP.STATS[1] = 0
        RP.STATS_BACK[0] = RP.STATS_BACK[1] = 0
        fold_s, m = one_fold()
        row = {"tag": tag, "arm": arm, "l1_fc1": want,
               "fold_s": round(fold_s, 3), "plddt": m.get("plddt"), "cif": sha_dir(struct_dir),
               "e6_served": RP.STATS_GATED[0], "e6_declined": RP.STATS_GATED[1],
               "l1_fc1_stats": list(EC.L1_FC1_STATS),
               "fwd_move": list(RP.STATS), "back_move": list(RP.STATS_BACK),
               "l1_out_refused": len(T._L1_OUT_REFUSED),
               "loadavg": open("/proc/loadavg").read().split()[0]}
        res["runs"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
        print("  %-14s %8.3fs plddt=%s e6=%d/%d l1fc1=%d/%d l1refused=%d cif=%s load=%s"
              % (tag, fold_s, m.get("plddt"), RP.STATS_GATED[0], RP.STATS_GATED[1],
                 EC.L1_FC1_STATS[0], EC.L1_FC1_STATS[1],
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
    if "base" in summary:
        b = summary["base"]["median"]
        for arm, s in summary.items():
            s["speedup_vs_base"] = round(b / s["median"], 4)
            s["delta_s"] = round(b - s["median"], 3)
        summary["A/A_noise_floor_s"] = summary["base"]["spread"]
    res["summary"] = summary
    cifs = {row["arm"]: tuple(sorted(row["cif"].items())) for row in res["runs"]}
    res["cif_identical_across_arms"] = len(set(cifs.values())) == 1
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print("CIF identical across all arms: %s" % res["cif_identical_across_arms"], flush=True)


if __name__ == "__main__":
    main()

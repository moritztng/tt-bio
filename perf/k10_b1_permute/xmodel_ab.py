#!/usr/bin/env python3
"""B1 is shared code, so check a sibling model's output does not move either.

`TriangleMultiplication` lives in tt_bio/tenstorrent.py and every ported model with a pair track
reaches it. The lever permutes that module's fused in-projection columns, so the bar on a sibling
is the same one Boltz-2 gets: the SAME structure file, byte for byte, not a parity band.

One process, one card, both arms per model, arms alternated so neither inherits the other's
weight cache. A model whose gated-move count is zero in both arms is reported as NOT REACHED
rather than as a pass -- an unchanged digest from code that never ran is not evidence.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="opendde")
    ap.add_argument("--recycles", type=int, default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RP
    import tt_baseline as B
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor

    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__
    assert "TT_BIO_TRIMUL_GP_BANK_SPLIT" not in os.environ, \
        "the arms are set in-process; an env pin would make both arms the same arm"
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "size": a.size, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_head": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
           "runs": [], "verdict": {}}
    bad = 0
    for model in [m for m in a.models.split(",") if m]:
        one_fold, meta, _state = B.build_fold(
            model, ROOT / (".msa_xmodel_%d" % a.size), a.fixdir / ("cdk2x2_%d.yaml" % a.size),
            a.fixdir / ("cdk2x2_%d.a3m" % a.size), recycling_steps=a.recycles)
        res["cfg"] = {"recycling_steps": meta["recycling_steps"],
                      "sampling_steps": B.SAMPLING_STEPS}
        struct_dir = Path(meta["struct_dir"])
        rows = {}
        for arm in ("off", "on", "off"):
            T.set_trimul_gp_bank_split(arm == "on")
            RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
            for p in struct_dir.glob("*"):
                if p.is_file():
                    p.unlink()
            fold_s, m = one_fold()
            row = {"model": model, "arm": arm, "roles": list(T.gp_roles()),
                   "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
                   "gated": list(RP.STATS_GATED), "cif": sha_dir(struct_dir)}
            res["runs"].append(row)
            rows.setdefault(arm, []).append(row)
            print("  %-12s %-3s %8.3fs gated=%s plddt=%s %s"
                  % (model, arm, fold_s, row["gated"], row["plddt"],
                     list(row["cif"].values())), flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
        reached = any(r["gated"][0] > 0 for r in res["runs"] if r["model"] == model)
        digests = {json.dumps(r["cif"], sort_keys=True) for r in res["runs"]
                   if r["model"] == model}
        v = ("NOT REACHED" if not reached else
             "EQUAL" if len(digests) == 1 else "DIFFERS")
        res["verdict"][model] = {"verdict": v, "n_digests": len(digests),
                                 "gated_calls": rows["on"][0]["gated"][0]}
        bad += v != "EQUAL"
        print("  -> %s %s" % (model, v), flush=True)
        a.out.write_text(json.dumps(res, indent=1))
    res["all_equal"] = bad == 0
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["verdict"], indent=1))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

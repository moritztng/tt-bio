#!/usr/bin/env python3
"""THE fold-level A/B for the esmfold2-to-4x integrated arm, at 512 aa.

Arms in ONE process against ONE built model, round-robin so host drift hits every arm equally.
`base` repeats once per round and its spread IS the A/A noise floor. Every lever is a runtime
switch, so no arm rebuilds weights and only the op sequence differs:

    base   the landed 37.871 s configuration (E6 on, split-fc1 on)
    armc   + the row-blocked L1-resident pair FFN (esmc.set_pair_l1_rows(32))

There is no in-projection arm: `perf/esm4x/mmcfg_sweep.py` swept all 71 legal
MinimalMatmulConfigs at [1,512,512,256] x [256,1024] and the best bit-exact one is 1.0169x,
under the 1.05x kill gate, so the lever was dropped before the build. Same for the channel
matmul's out_subblock, where the shipped 1x1 wins at the production shape.

Acceptance per arm: CIF sha256 295867277b9c137f, plDDT 0.9285, benchlock held, loadavg recorded.
"""
import argparse, hashlib, json, os, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

ARMS = {"base": 0, "armc": 32}      # arm -> pair FFN row height, 0 = the landed full-size path


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="base,armc")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--screen", type=Path,
                    help="mmcfg_sweep json, recorded alongside the A/B it decided")
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
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "arms": arms, "rounds": a.rounds,
           "runs": []}

    if a.screen:
        sc = json.loads(a.screen.read_text())
        res["inproj_screen_best"] = (sc["A_inproj"]["L512"].get("best") or [None])[0]
        res["subblock_screen"] = sc["B_subblock"]

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_om512_{a.size}", tgt, a3m)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    struct_dir = Path(meta["struct_dir"])

    def run(tag, arm):
        rows = ARMS[arm]
        EC.set_pair_l1_rows(rows)
        RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
        fold_s, m = one_fold()
        row = {"tag": tag, "arm": arm, "pair_l1_rows": rows,
               "fold_s": round(fold_s, 3), "plddt": m.get("plddt"), "cif": sha_dir(struct_dir),
               "e6_served": RP.STATS_GATED[0], "e6_declined": RP.STATS_GATED[1],
               "l1_out_refused": len(T._L1_OUT_REFUSED),
               "loadavg": open("/proc/loadavg").read().split()[0]}
        res["runs"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {tag:16s} {fold_s:8.3f}s plddt={m.get('plddt')} "
              f"e6={RP.STATS_GATED[0]}/{RP.STATS_GATED[1]} l1refused={row['l1_out_refused']} "
              f"cif={list(row['cif'].values())[0] if row['cif'] else '-'} "
              f"load={row['loadavg']}", flush=True)
        return fold_s

    print(f"=== {a.model} {a.size} aa rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS} "
          f"arms={arms} rounds={a.rounds} ===", flush=True)
    for arm in arms:
        run(f"warm:{arm}", arm)
    for r in range(a.rounds):
        for arm in arms:
            run(f"r{r}:{arm}", arm)

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
            s["x_vs_h200"] = round(s["median"] / 7.256, 4)
        summary["A/A_noise_floor_s"] = summary["base"]["spread"]
    res["summary"] = summary
    cifs = {row["arm"]: tuple(sorted(row["cif"].items())) for row in res["runs"]}
    res["cif_identical_across_arms"] = len(set(cifs.values())) == 1
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print(f"CIF identical across all arms: {res['cif_identical_across_arms']}", flush=True)


if __name__ == "__main__":
    main()

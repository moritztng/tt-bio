#!/usr/bin/env python3
"""The one fold-level A/B for ESMFold2 at 512 aa: E6 and the split-fc1 SwiGLU, alone and together.

Four arms in ONE process against ONE built model, interleaved round-robin so host drift hits
every arm equally. `base` repeats once per round and its spread IS the A/A noise floor -- there is
no separate A/A run to drift away from. Both levers are runtime switches
(`reblock_permute.set_enabled_gated`, `esmc.set_split_swiglu`), so no arm rebuilds weights and the
only thing that differs between arms is the op sequence.

E6's served/declined counters are recorded per fold: an arm that serves 0 moves is an arm that
was never tested, and this lineage has published one of those before.
"""
import argparse, hashlib, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

ARMS = {"base": (False, False), "e6": (True, False),
        "split": (False, True), "both": (True, True)}


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="base,e6,split,both")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    # `build_fold` takes `fast` and this script never passed it, so esmfold2 could only be
    # folded here in normal precision. On Wormhole that is not a choice: the ESMC-6B LM is
    # ~12.8 GB against a ~12 GB chip, and the forcing lives in main.py's CLI path, which
    # build_fold does not go through. A --fast arm compares only to another --fast arm.
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]

    import ttnn
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
           "arms": arms, "rounds": a.rounds, "runs": []}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_om512_{a.size}", tgt, a3m, fast=a.fast)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    struct_dir = Path(meta["struct_dir"])

    def run(tag, arm):
        e6, split = ARMS[arm]
        RP.set_enabled_gated(e6)
        EC.set_split_swiglu(split)
        RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
        fold_s, m = one_fold()
        row = {"tag": tag, "arm": arm, "e6": e6, "split": split,
               "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "cif": sha_dir(struct_dir),
               "e6_served": RP.STATS_GATED[0], "e6_declined": RP.STATS_GATED[1],
               "loadavg": open("/proc/loadavg").read().split()[0]}
        res["runs"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {tag:16s} {fold_s:8.3f}s plddt={m.get('plddt')} "
              f"e6={RP.STATS_GATED[0]}/{RP.STATS_GATED[1]} "
              f"cif={list(row['cif'].values())[0] if row['cif'] else '-'} "
              f"load={row['loadavg']}", flush=True)
        return fold_s

    print(f"=== {a.model} {a.size} aa rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS} "
          f"arms={arms} rounds={a.rounds} ===", flush=True)
    # Every arm compiles its own program variants on first use; warm each once so no round pays
    # a JIT cost the others do not. `warm:base` is also the cold fold.
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
        summary[arm] = {"n": len(v), "median": round(st.median(v), 3),
                        "min": min(v), "max": max(v), "spread": round(max(v) - min(v), 3)}
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
    print(f"CIF identical across all arms: {res['cif_identical_across_arms']}", flush=True)


if __name__ == "__main__":
    main()

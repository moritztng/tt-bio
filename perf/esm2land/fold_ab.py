#!/usr/bin/env python3
"""Fold-level A/B for the two pair-transition levers, in one process against one built model.

Arms, round-robin so host drift hits every arm equally. `base` runs once per round and its spread
IS the A/A noise floor. Both levers are runtime switches, so no arm rebuilds weights and only the
op sequence differs:

    base    what main shipped before this branch: unsplit fc1 -> chunk -> silu -> multiply
    armA    split fc1 + SiLU as the multiply's operand-A activation
    armAC   armA + the row-blocked SwiGLU product in L1

Acceptance: CIF sha256 and plDDT identical across every arm at every size (both levers are
torch.equal at the op), and any timing delta at least 3x the A/A spread. At an L that 32 does not
divide, armAC's gate declines by design, so armAC must land inside armA's spread there.

Also counts, per fold, how many SwiGLUFFN instances actually took the split path. That is the
cross-model check: it must equal the PairUpdateBlock count for esmfold2 and be 0 for ESMC.
"""
import argparse, hashlib, json, os, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

# arm -> (split_swiglu, pair FFN row height)
ARMS = {"base": (False, 0), "armA": (True, 0), "armAC": (True, 32)}


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="base,armA,armAC")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]

    import tt_bio.tenstorrent as T
    import tt_bio.esmc as EC
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    # Which SwiGLUFFN instances take the split path, counted at the call not read off the flag.
    seen = {}
    orig_call = EC.SwiGLUFFN.__call__

    def counted(self, x):
        key = id(self)
        e = seen.setdefault(key, {"split": bool(self.split_swiglu), "calls": 0, "ndim": len(x.shape)})
        e["calls"] += 1
        return orig_call(self, x)

    EC.SwiGLUFFN.__call__ = counted

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "arms": arms, "rounds": a.rounds, "runs": []}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_om512_{a.size}", tgt, a3m)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    res["split_swiglu_default"] = EC.SPLIT_SWIGLU
    res["pair_ffn_row_block_default"] = EC.PAIR_FFN_ROW_BLOCK
    struct_dir = Path(meta["struct_dir"])

    def run(tag, arm):
        split, rows = ARMS[arm]
        EC.set_split_swiglu(split)
        EC.set_pair_ffn_row_block(rows)
        seen.clear()
        fold_s, m = one_fold()
        n_split = sum(1 for e in seen.values() if e["split"])
        row = {"tag": tag, "arm": arm, "split_swiglu": split, "pair_ffn_row_block": rows,
               "fold_s": round(fold_s, 3), "plddt": m.get("plddt"), "cif": sha_dir(struct_dir),
               "ffn_instances": len(seen), "ffn_instances_split": n_split,
               "loadavg": open("/proc/loadavg").read().split()[0]}
        res["runs"].append(row)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {tag:14s} {fold_s:8.3f}s plddt={m.get('plddt')} "
              f"ffn={n_split}/{len(seen)} split "
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
        if not row["tag"].startswith("warm:"):
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
    res["plddt_identical_across_arms"] = len({r["plddt"] for r in res["runs"]}) == 1
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print(f"CIF identical across all arms: {res['cif_identical_across_arms']}", flush=True)
    print(f"plDDT identical across all arms: {res['plddt_identical_across_arms']}", flush=True)


if __name__ == "__main__":
    main()

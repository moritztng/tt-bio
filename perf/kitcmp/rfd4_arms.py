"""Drive every RFD4 B200 arm under state/rfd4/PROTOCOL.md section 8, rank them, re-run the winner clean.

Each (arm, size) is its own `rfd4_gpu.py` process, so a compile crash or an OOM costs that arm and not the
session, and no arm inherits another's compile cache. Every process appends to its own runs.jsonl under
`--out/<arm>/<size>/`; this file only schedules, ranks and writes `--out/ARMS.json`.

    python rfd4_arms.py --inputs DIR --out DIR --deadline EPOCH [--screen-warm 5] [--final-warm 9]
    python rfd4_arms.py --inputs DIR --out DIR --deadline EPOCH --reference 30

An arm that fails is kept in the ranking with `"status": "FAILED"` and its error, never dropped: a silent drop
once nearly settled "NVIDIA's best" on an arm without the accelerator the model ships with (BC4, 2026-10-19).
"""
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = (128, 256, 512)

# name -> Hydra overrides. Same work in every one of them; see PROTOCOL.md section 1 for what is NOT an arm.
ARMS = {
    "native-eager":           ["compile_lengths=null"],
    "native-compile":         [],
    "compile-reduce-overhead": ["nn.torch_compile_mode=reduce-overhead"],
    "compile-cueq":           ["nn.tri_mult_backend=cuequivariance", "nn.attention_type=cuequivariance"],
    "attn-torch":             ["nn.tri_mult_backend=cuequivariance", "nn.attention_type=torch"],
    "bioir":                  ["inference_backend=bioir"],
    # the 512 bucket is above bioir's default CUDA-graph ceiling (511), so the graph arm is explicit
    "bioir-graph512":         ["inference_backend=bioir", "bioir.cuda_graph_max_tokens=1024"],
}
SCREEN_BATCH = {128: 8, 256: 4, 512: 2}
SWEEP = {128: (1, 2, 4, 8, 16), 256: (1, 2, 4, 8), 512: (1, 2, 4)}


def run(arm, size, batch, warm, overrides, tag=None, spec=None, extra=()):
    """One rfd4_gpu.py process. Returns its warm records.

    A compiled arm compiles only the bucket it runs: the shipped `compile_lengths` is 15 buckets from 128 to
    1024, and compiling 14 shapes a run never uses would inflate its cold line without changing a warm one.
    Both sides still pad to the same bucket, so this is the same work, measured without the dead weight.
    """
    if not any(o.startswith(("compile_lengths=", "inference_backend=")) for o in list(overrides) + list(extra)):
        overrides = list(overrides) + [f"compile_lengths=[{size}]"]
    out = a.out / (tag or arm) / f"{size}-b{batch}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "rfd4_gpu.py"), "--spec", str(spec or a.inputs / f"pdl1-{size}.yaml"),
           "--size", str(size), "--arm", arm, "--batch", str(batch), "--warm", str(warm), "--out", str(out)]
    for o in list(overrides) + list(extra):
        cmd += ["--set", o]
    print(f"{time.strftime('%FT%TZ', time.gmtime())} -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, timeout=max(60, a.deadline - int(time.time())), check=False)
    recs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()] if (out / "runs.jsonl").exists() else []
    return recs


def summarise(recs):
    warm = [r for r in recs if r.get("kind") == "warm" and r.get("status") != "FAILED" and r["wall_s"] > 0]
    failed = [r for r in recs if r.get("status") == "FAILED"]
    if not warm:
        err = failed[0].get("error") if failed else "no warm batch finished"
        return {"status": "FAILED", "error": err, "n": 0}
    rate = sorted(r["designs"] / r["wall_s"] for r in warm)
    mhz = [r["sm_mhz"] for r in warm if r.get("sm_mhz")]
    return {"status": "ok", "n": len(rate), "designs_s": statistics.median(rate),
            "spread": [rate[0], rate[-1]], "batch": warm[0]["batch"],
            "sm_mhz_median": statistics.median(mhz) if mhz else None}


def time_left(margin=600):
    return time.time() < a.deadline - margin


def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deadline", required=True, type=int)
    ap.add_argument("--screen-warm", type=int, default=5)
    ap.add_argument("--final-warm", type=int, default=9)
    ap.add_argument("--reference", type=int, default=0, help="reference-design mode: N designs per feature family")
    a = ap.parse_args()

    if a.reference:
        # The chosen arm, the documented 50-120 binder range, scored: rfd4-quality's reference distribution.
        chosen = json.loads((a.out / "ARMS.json").read_text())["best_arm"]
        ov = ARMS[chosen] + ["metrics_tags='[ppi_refolding,hotspot_conditioning]'"]
        done = {}
        for fam, spec in sorted({p.stem: p for p in a.inputs.glob("pdl1-ref*.yaml")}.items()):
            if not time_left(1200):
                break
            recs = run(chosen, 256, min(8, a.reference), a.reference // min(8, a.reference), ov,
                       tag="reference", spec=spec)
            done[fam] = {"designs": sum(r.get("designs", 0) for r in recs), "spec": str(spec), "arm": chosen}
        (a.out / "REFERENCE.json").write_text(json.dumps(done, indent=2) + "\n")
        print(json.dumps(done, indent=2))
        sys.exit(0)

    results = {}
    for size in SIZES:
        for arm, ov in ARMS.items():
            if not time_left():
                print(f"deadline: skipping {arm} @ {size}", flush=True)
                continue
            if arm == "bioir-graph512" and size != 512:
                continue
            results.setdefault(str(size), {})[arm] = summarise(
                run(arm, size, SCREEN_BATCH[size], a.screen_warm, ov))
            (a.out / "ARMS.json").write_text(json.dumps({"screen": results}, indent=2) + "\n")

    # winner per size from the screen, then its batch sweep, then the clean final run at --final-warm
    best, final = {}, {}
    for size in SIZES:
        ok = {k: v for k, v in results.get(str(size), {}).items() if v["status"] == "ok"}
        if not ok:
            continue
        arm = max(ok, key=lambda k: ok[k]["designs_s"])
        sweep = {}
        for b in SWEEP[size]:
            if not time_left():
                break
            sweep[b] = summarise(run(arm, size, b, max(3, a.screen_warm - 2), ARMS[arm], tag=f"sweep-{arm}"))
        okb = {b: v for b, v in sweep.items() if v["status"] == "ok"}
        bb = max(okb, key=lambda b: okb[b]["designs_s"]) if okb else SCREEN_BATCH[size]
        best[str(size)] = {"arm": arm, "batch": bb, "sweep": {str(k): v for k, v in sweep.items()}}
        if time_left():
            final[str(size)] = summarise(run(arm, size, bb, a.final_warm, ARMS[arm], tag=f"final-{arm}"))
        (a.out / "ARMS.json").write_text(json.dumps(
            {"screen": results, "best": best, "final": final,
             "best_arm": max(best.values(), key=lambda v: 0)["arm"] if best else None}, indent=2) + "\n")

    # one chosen arm across sizes: the one that wins most sizes, ties broken by the largest size's rate
    if best:
        votes = {}
        for s, v in best.items():
            votes[v["arm"]] = votes.get(v["arm"], 0) + 1
        chosen = max(votes, key=lambda k: (votes[k], final.get("512", {}).get("designs_s", 0) if best.get("512", {}).get("arm") == k else 0))
        (a.out / "ARMS.json").write_text(json.dumps(
            {"screen": results, "best": best, "final": final, "best_arm": chosen}, indent=2) + "\n")
        print(f"best_arm={chosen}")


if __name__ == "__main__":
    main()

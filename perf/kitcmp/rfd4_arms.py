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
# The GPU is entitled to its own best batch (STANDING memory gpu-comparison-must-be-one-protocol-both-sides).
# A fixed ladder stopping at 16/8/4 is not "its best": at 512 x b2 the B200 held 8.5 GiB of ~180 GiB and the
# rate was still climbing at the top rung, so the ladder would have set NVIDIA's number instead of the card.
# Double until the card actually pushes back -- a failed/OOM rung, or two rungs that buy nothing.
SWEEP_CEIL = {128: 128, 256: 64, 512: 32}


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


def climb(arm, size, warm, tag, seed=None):
    """Double the batch until the card pushes back. Returns {batch: summary}.

    Stops on the first rung that fails (OOM is the expected failure) or after two rungs that do not beat the
    best rate by more than 2 %, so a card with headroom keeps climbing and one without stops paying for it.
    `seed` carries rungs an earlier sweep on this same box already measured, so extending a capped ladder
    pays only for the rungs above it.
    """
    seed = {int(k): v for k, v in (seed or {}).items()}
    sweep, best_rate, stall, b = {}, 0.0, 0, 1
    while b <= SWEEP_CEIL[size] and time_left():
        if b in seed:
            s = seed[b]
            print(f"climb {arm}@{size}: b{b} reused from this box's earlier sweep ({s.get('status')})",
                  flush=True)
        else:
            s = summarise(run(arm, size, b, warm, ARMS[arm], tag=tag))
        sweep[b] = s
        if s["status"] != "ok":
            print(f"climb {arm}@{size}: b{b} failed ({s.get('error')}) -- that is the ceiling", flush=True)
            break
        if s["designs_s"] > best_rate * 1.02:
            best_rate, stall = s["designs_s"], 0
        else:
            stall += 1
            if stall >= 2:
                print(f"climb {arm}@{size}: two rungs with no gain, stopping at b{b}", flush=True)
                break
        b *= 2
    else:
        if b > SWEEP_CEIL[size] and stall == 0:
            print(f"climb {arm}@{size}: hit the configured ceiling b{SWEEP_CEIL[size]} while still gaining "
                  f"-- the ceiling, not the card, bounded this", flush=True)
    return sweep


def choose_arm(best, final):
    """One arm across sizes: the one that wins most sizes, ties broken by the largest size's rate.

    Used for the reference designs, so it must be right in every partial write of ARMS.json too -- a run that
    dies after 256 still has to name the arm its own numbers support.
    """
    if not best:
        return None
    votes = {}
    for v in best.values():
        votes[v["arm"]] = votes.get(v["arm"], 0) + 1
    big = max(best, key=int)
    return max(votes, key=lambda k: (votes[k],
                                     final.get(big, {}).get("designs_s", 0) if best[big]["arm"] == k else 0))


def main():
    global a
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deadline", required=True, type=int)
    ap.add_argument("--screen-warm", type=int, default=5)
    ap.add_argument("--final-warm", type=int, default=9)
    ap.add_argument("--reference", type=int, default=0, help="reference-design mode: N designs per feature family")
    ap.add_argument("--reference-arm", help="arm to generate reference designs with (default: ARMS.json best_arm). "
                                            "The extended climb can re-rank, and reference designs must come "
                                            "from the arm actually published, not the screen's winner.")
    ap.add_argument("--sweep-only", action="store_true",
                    help="re-climb the batch of each size's top arms from ARMS.json, without the screen")
    ap.add_argument("--arms", type=int, default=2,
                    help="how many arms per size the extended climb re-ranks (default 2)")
    a = ap.parse_args()

    if a.sweep_only:
        # Session 3 screened and swept under the old 16/8/4 ladder, which bounded the B200 below its own best.
        # This re-climbs the winners on the same box, in the same session, so the published number is the
        # card's ceiling and not the ladder's. Writes beside ARMS.json, never over it.
        prev = json.loads((a.out / "ARMS.json").read_text())
        ext = {}
        for size in SIZES:
            ok = {k: v for k, v in prev.get("screen", {}).get(str(size), {}).items() if v["status"] == "ok"}
            if not ok or not time_left():
                continue
            # The screen ranked every arm at ONE batch, and that batch left the card 22-50 % busy, so the
            # ranking is a ranking of host overhead as much as of the arms. The most host-bound arm has the
            # most to gain from a bigger batch, so re-rank the top --arms arms, each at its own best batch,
            # rather than freezing NVIDIA's choice on the screen's winner.
            ranked = sorted(ok, key=lambda k: ok[k]["designs_s"], reverse=True)[:max(1, a.arms)]
            per_arm = {}
            for arm in ranked:
                if not time_left():
                    break
                was = prev.get("best", {}).get(str(size), {})
                seed = was.get("sweep") if was.get("arm") == arm else None
                sweep = climb(arm, size, max(3, a.screen_warm - 2), f"ext-{arm}", seed=seed)
                okb = {b: v for b, v in sweep.items() if v["status"] == "ok"}
                if not okb:
                    per_arm[arm] = {"sweep": {str(k): v for k, v in sweep.items()}}
                    continue
                bb = max(okb, key=lambda b: okb[b]["designs_s"])
                per_arm[arm] = {"batch": bb, "designs_s": okb[bb]["designs_s"],
                                "sweep": {str(k): v for k, v in sweep.items()}}
                (a.out / "ARMS-EXT.json").write_text(json.dumps(ext | {str(size): {"arms": per_arm}}, indent=2) + "\n")
            rated = {k: v for k, v in per_arm.items() if v.get("designs_s")}
            ext[str(size)] = {"arms": per_arm}
            if rated:
                win = max(rated, key=lambda k: rated[k]["designs_s"])
                ext[str(size)] |= {"arm": win, "batch": rated[win]["batch"]}
                if time_left():
                    ext[str(size)]["final"] = summarise(
                        run(win, size, rated[win]["batch"], a.final_warm, ARMS[win], tag=f"extfinal-{win}"))
            (a.out / "ARMS-EXT.json").write_text(json.dumps(ext, indent=2) + "\n")
        print(json.dumps(ext, indent=2))
        sys.exit(0)

    if a.reference:
        # The chosen arm, the documented 50-120 binder range, scored: rfd4-quality's reference distribution.
        chosen = a.reference_arm or json.loads((a.out / "ARMS.json").read_text())["best_arm"]
        if chosen not in ARMS:
            sys.exit(f"reference: unknown arm {chosen!r}")
        print(f"reference designs from arm {chosen}", flush=True)
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

    # The screen ranks every arm at one small batch, where the card is far from busy, so it partly ranks host
    # overhead rather than the arms. Climb the top --arms arms per size and rank them at each arm's own best
    # batch; the ranking can invert, and the reference designs are generated from whatever wins here.
    best, final = {}, {}
    for size in SIZES:
        ok = {k: v for k, v in results.get(str(size), {}).items() if v["status"] == "ok"}
        if not ok:
            continue
        per_arm = {}
        for arm in sorted(ok, key=lambda k: ok[k]["designs_s"], reverse=True)[:a.arms]:
            sweep = climb(arm, size, max(3, a.screen_warm - 2), f"sweep-{arm}")
            okb = {b: v for b, v in sweep.items() if v["status"] == "ok"}
            if okb:
                bb = max(okb, key=lambda b: okb[b]["designs_s"])
                per_arm[arm] = {"batch": bb, "designs_s": okb[bb]["designs_s"],
                                "sweep": {str(k): v for k, v in sweep.items()}}
        if not per_arm:
            continue
        arm = max(per_arm, key=lambda k: per_arm[k]["designs_s"])
        bb = per_arm[arm]["batch"]
        best[str(size)] = {"arm": arm, "batch": bb, "climbed": per_arm,
                           "sweep": per_arm[arm]["sweep"]}
        if time_left():
            final[str(size)] = summarise(run(arm, size, bb, a.final_warm, ARMS[arm], tag=f"final-{arm}"))
        (a.out / "ARMS.json").write_text(json.dumps(
            {"screen": results, "best": best, "final": final,
             "best_arm": choose_arm(best, final)}, indent=2) + "\n")

    if best:
        chosen = choose_arm(best, final)
        (a.out / "ARMS.json").write_text(json.dumps(
            {"screen": results, "best": best, "final": final, "best_arm": chosen}, indent=2) + "\n")
        print(f"best_arm={chosen}")


if __name__ == "__main__":
    main()

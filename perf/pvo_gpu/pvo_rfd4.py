"""RFD4-Proteina on the rented A100-40, for PVO: price per design, then the designs pvo-accuracy grades.

One target (state/pvo/INPUTS.md: 9TH6's antigen whole, 448 residues + a 120-residue binder = 568 tokens, a
620-wide sample in the 624 compile bucket), modes from state/pvo/MODES.md. Exact is `native-compile` at b1, which is what a Modal user
pays per design today. Fast is whichever shipped arm wins at its own best batch on THIS card: the published
fast arms are a B200 ranking and batch is RFD4's dominant lever, so it is re-swept here rather than assumed.

    python pvo_rfd4.py --spec DIR/9TH6-A-b120.yaml --out DIR --deadline EPOCH [--designs 128]

Scheduling only. Every arm is its own `rfd4_gpu.py` process (one compile cache, one failure blast radius) and
every rung lands in RFD4.json as it finishes, so a box that dies keeps what it measured.
"""
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# On the box every script is copied flat into /root/kc; in the repo rfd4_gpu.py lives under perf/kitcmp.
RFD4_GPU = HERE / "rfd4_gpu.py"
if not RFD4_GPU.exists():
    RFD4_GPU = HERE.parent / "kitcmp" / "rfd4_gpu.py"
# The contig `120-120, /0, A25-524` spans 500 target positions (448 of them modelled, 52 unmodelled gaps the
# datapipe still carries) plus the 120-residue binder, so RFD4's sample dimension is 620, not 568. Session 6 asked
# for 576 and every arm died in the collate with "Sample dimension I=620 exceeds the largest bucket 576". 620 is
# also the sample dim pvo-tt ran on Wormhole (state/pvo/tt/wh-rfd4/fair.jsonl "size": 620), so both sides do the
# same work. The compile bucket holding it is 624: upstream asserts compile_lengths are 16-aligned (session 8 asked
# for 620 and the harness refused it).
SIZE = 624
# MODES.md: exact is upstream's shipped config, fast is every same-work switch upstream ships. The two compiled
# arms below are the ones that won a size on the B200 (state/rfd4/FAIR.md); native-eager, bioir and the graph
# variants were rejected on speed there and are not re-paid for here.
ARMS = {
    "native-compile": [],
    "compile-cueq": ["nn.tri_mult_backend=cuequivariance", "nn.attention_type=cuequivariance"],
    "compile-reduce-overhead": ["nn.torch_compile_mode=reduce-overhead"],
}
EXACT = "native-compile"
CEIL = 64        # 568 tokens in 40 GB; the climb normally stops on an OOM long before this
DESIGN_SEED = 1234


def run(arm, batch, warm, tag, seed=None, designs_only=False):
    out = a.out / tag / f"b{batch}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(RFD4_GPU), "--spec", str(a.spec), "--size", str(SIZE),
           "--arm", arm, "--batch", str(batch), "--warm", str(warm), "--out", str(out),
           # One bucket: the shipped compile_lengths is 15 shapes and the run uses one of them.
           "--set", f"compile_lengths=[{SIZE}]"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    for o in ARMS[arm]:
        cmd += ["--set", o]
    print(f"{time.strftime('%FT%TZ', time.gmtime())} -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, timeout=max(60, a.deadline - int(time.time())), check=False)
    f = out / "runs.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def summarise(recs):
    warm = [r for r in recs if r.get("kind") == "warm" and r.get("status") != "FAILED" and r["wall_s"] > 0]
    failed = [r for r in recs if r.get("status") == "FAILED"]
    if not warm:
        return {"status": "FAILED", "error": failed[0].get("error") if failed else "no warm batch finished", "n": 0}
    rate = sorted(r["designs"] / r["wall_s"] for r in warm)
    mhz = [r["sm_mhz"] for r in warm if r.get("sm_mhz")]
    cold = next((r for r in recs if r.get("kind") == "cold" and r.get("phase") == "load"), None)
    return {"status": "ok", "n": len(rate), "designs_s": statistics.median(rate), "s_per_design": 1 / statistics.median(rate),
            "spread": [rate[0], rate[-1]], "batch": warm[0]["batch"],
            "load_s": cold["wall_s"] if cold else None,
            "sm_mhz_median": statistics.median(mhz) if mhz else None}


def time_left(margin=600):
    return time.time() < a.deadline - margin


def save():
    (a.out / "RFD4.json").write_text(json.dumps(R, indent=2) + "\n")


def climb(arm):
    """Double the batch until the card pushes back: a failed rung (OOM is the expected one) or two rungs
    that buy under 2 %. The GPU is entitled to its own best batch, so nothing caps this but 40 GB."""
    sweep, best, stall, b = {}, 0.0, 0, 1
    while b <= CEIL and time_left():
        s = summarise(run(arm, b, 3, f"sweep-{arm}"))
        sweep[str(b)] = s
        R["sweeps"][arm] = sweep
        save()
        if s["status"] != "ok":
            print(f"{arm}: b{b} failed ({s.get('error')}) -- that is the ceiling", flush=True)
            break
        if s["designs_s"] > best * 1.02:
            best, stall = s["designs_s"], 0
        else:
            stall += 1
            if stall >= 2:
                print(f"{arm}: two rungs with no gain, stopping at b{b}", flush=True)
                break
        b *= 2
    return sweep


def best_rung(sweep):
    ok = {int(b): v for b, v in sweep.items() if v["status"] == "ok"}
    return max(ok, key=lambda b: ok[b]["designs_s"]) if ok else None


def designs(arm, tag, n):
    """n designs from `arm` at a batch it is known to survive, seeds DESIGN_SEED+i. Exact and fast use the
    same batch and the same seeds, so pvo-accuracy's comparison is paired (state/pvo/RFD4-ACC.md)."""
    d = a.out / tag
    if sum(1 for _ in d.rglob("*_generated.cif*")) >= n:
        print(f"designs {tag}: already have {n}", flush=True)
        return
    batch = R.get("design_batch") or 1
    recs = run(arm, batch, -(-n // batch) - 1, tag, seed=DESIGN_SEED)
    made = sum(r.get("designs", 0) for r in recs if r.get("status") != "FAILED")
    R["designs"][tag] = {"arm": arm, "batch": batch, "made": made, "wanted": n,
                         "seeds": [DESIGN_SEED + i for i in range(-(-n // batch))]}
    save()
    print(f"designs {tag}: {made}/{n}", flush=True)


def main():
    global a, R
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deadline", required=True, type=int)
    ap.add_argument("--designs", type=int, default=128, help="designs per mode for pvo-accuracy (0 = timing only)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    R = {"size": SIZE, "spec": str(a.spec), "sweeps": {}, "designs": {}}
    save()

    # Exact first: b1 is the per-design price, and it is the number the proposal cannot do without.
    R["exact_serial"] = summarise(run(EXACT, 1, 3, "exact-b1"))
    save()
    if R["exact_serial"]["status"] != "ok":
        print(f"exact b1 FAILED ({R['exact_serial'].get('error')}) -- 568 tokens may not fit this card", flush=True)

    for arm in ARMS:
        if time_left():
            climb(arm)

    rated = {arm: best_rung(sw) for arm, sw in R["sweeps"].items()}
    rated = {arm: b for arm, b in rated.items() if b}
    if rated:
        win = max(rated, key=lambda arm: R["sweeps"][arm][str(rated[arm])]["designs_s"])
        R["fast_arm"], R["fast_batch"] = win, rated[win]
        R["fast"] = R["sweeps"][win][str(rated[win])]
        # The design arms are paired by seed, so both modes run at one batch: the widest batch exact itself
        # survived, capped at 16 so a single arm cannot eat the whole remaining window.
        R["design_batch"] = min(16, best_rung(R["sweeps"].get(EXACT, {})) or 1)
        save()
        print(f"fast arm {win} b{rated[win]} {R['fast']['s_per_design']:.2f} s/design "
              f"(exact b1 {R['exact_serial'].get('s_per_design', float('nan')):.2f})", flush=True)

    if a.designs and R.get("fast_arm") and time_left(1800):
        designs(EXACT, "designs-exact", a.designs)
        if time_left(1800):
            designs(R["fast_arm"], "designs-fast", a.designs)
        else:
            print("deadline: the fast design arm did not run", flush=True)
    save()
    print(json.dumps({k: v for k, v in R.items() if k != "sweeps"}, indent=2))


if __name__ == "__main__":
    main()

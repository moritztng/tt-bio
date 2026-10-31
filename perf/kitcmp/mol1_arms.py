"""Every MOL1 B200 arm (state/mol1/PROTOCOL.md section 6), in an order that leaves the most valuable cells behind
when the box is cut short.

    python mol1_arms.py --inputs /root/kc/inputs --out /root/kc/mol1 --deadline <epoch> [--warm 5]

1. the four kernel arms at 512 (upstream's own kernel table is measured at this size);
2. +tf32 and +chunk5 on the 512 winner, kept only if faster;
3. the best stack and the runner-up kernel arm at 256 and 1024;
4. concurrency K = 2, 4, ... on the best stack per size until a rung fails or buys under 2 %.
Each cell is one `mol1_gpu.py` process; its runs.jsonl lines land in <out>/<cell>/. ARMS.json is rewritten after
every cell, so the summary on disk is always current. A cell that fails is recorded FAILED with its error.
"""
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path

KERNELS = {"cueq-cueq": ("cuequivariance", "cuequivariance"), "cueq-triatt": ("cuequivariance", "triattention"),
           "fpf-cueq": ("flashpairformer", "cuequivariance"), "fpf-triatt": ("flashpairformer", "triattention")}
# Rough warm seconds per prediction at K=1, used only to skip a cell that cannot finish before the deadline.
EST_S = {256: 25, 512: 60, 1024: 150}


def rate(cell_dir):
    rs = [json.loads(l) for l in (cell_dir / "runs.jsonl").read_text().splitlines()] if (cell_dir / "runs.jsonl").exists() else []
    w = [r["predictions"] / r["wall_s"] for r in rs if r["kind"] == "warm" and r.get("status") != "FAILED" and r["wall_s"] > 0]
    err = next((r.get("error") for r in rs if r.get("status") == "FAILED"), None)
    return (statistics.median(w) if w else 0.0), len(w), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deadline", required=True, type=int)
    ap.add_argument("--warm", type=int, default=5)
    a = ap.parse_args()
    arms = {}

    def cell(size, name, kern, k=1, tf32=False, chunk5=False):
        key = f"{size}/{name}/k{k}"
        need = EST_S[size] * k * (a.warm + 2) * 1.3 + 120
        if time.time() + need > a.deadline:
            arms[key] = {"status": "SKIPPED", "reason": f"needs ~{need:.0f}s, past the deadline"}
        else:
            d = a.out / f"{size}-{name}-k{k}"
            cmd = [sys.executable, str(Path(__file__).with_name("mol1_gpu.py")), "--input", str(a.inputs / f"mol1-{size}.json"),
                   "--size", str(size), "--arm", name, "--k", str(k), "--warm", str(a.warm), "--out", str(d),
                   "--trimul", kern[0], "--triatt", kern[1]] + (["--tf32"] if tf32 else []) + (["--chunk5"] if chunk5 else [])
            d.mkdir(parents=True, exist_ok=True)
            with (d.parent / f"{d.name}.log").open("w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            r, n, err = rate(d)
            arms[key] = {"predictions_per_s": round(r, 5), "n": n, "rc": rc, "dir": str(d),
                         **({"status": "FAILED", "error": (err or f"rc {rc}")[-600:]} if n == 0 else {})}
        (a.out / "ARMS.json").write_text(json.dumps(arms, indent=1))
        return arms[key].get("predictions_per_s", 0.0)

    # 1. kernel arms at 512
    r512 = {n: cell(512, n, kern) for n, kern in KERNELS.items()}
    order = sorted(r512, key=r512.get, reverse=True)
    best, second = order[0], order[1]
    # 2. same-work options on the winner, kept only when they pay
    stack, base = {"name": best, "kern": KERNELS[best], "tf32": False, "chunk5": False}, r512[best]
    for opt in ("chunk5", "tf32"):
        trial = {**stack, opt: True}
        trial["name"] = stack["name"] + f"+{opt}"
        r = cell(512, trial["name"], trial["kern"], tf32=trial["tf32"], chunk5=trial["chunk5"])
        if r > base:
            stack, base = trial, r
    S = lambda size, k=1: cell(size, stack["name"], stack["kern"], k, stack["tf32"], stack["chunk5"])
    # 3. other sizes: the stack, and the runner-up kernel arm so the ranking is re-checked per size
    best_k = {512: (1, base)}
    for size in (256, 1024):
        best_k[size] = (1, S(size))
        cell(size, second, KERNELS[second])
    # 4. concurrency climb per size, largest payoff first
    for size in (512, 256, 1024):
        k, r = best_k[size]
        while r > 0:
            r2 = S(size, k * 2)
            if r2 < r * 1.02:
                break
            k, r = k * 2, r2
        best_k[size] = (k, r)
    summary = {"stack": stack, "best": {s: {"k": k, "predictions_per_s": round(r, 5)} for s, (k, r) in best_k.items()}}
    (a.out / "ARMS.json").write_text(json.dumps({**arms, "_summary": summary}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

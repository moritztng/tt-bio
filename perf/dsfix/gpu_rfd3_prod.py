"""RFD3 production benchmark on one GPU: the pinned R4 fixture, shipped settings, warm median.

Unlike `gpu_rfd3_sweep.py` (which measured a step differential across a size ladder to pick a
fixture) this runs the real production job -- 200 timesteps at the shipped batch -- and reports
seconds per design. No differential, no extrapolation: the number is the wall of a batch that
actually wrote its designs, divided by the designs in it.

Per point one process runs `n_batches` batches, so model load is paid once. The engine logs
"Finished inference batch in X seconds" per batch (rfd3/engine.py), wrapping the sampler loop only.
Batch 0 is discarded as cold, the rest are the warm sample. Power is sampled at 200 ms and reduced
over the warm window alone, located by the engine's own log stamps, because model load and the cold
batch sit near idle and drag the median down.

Every point is validated before its timing is kept: the designs must exist, the count must equal
the batch, coordinates must be finite, the residue topology must match the reference (585-residue
target plus 100-residue designed binder), and the seed written into the output metadata must be the
seed asked for. A run that exits 0 without writing structures is a failure here, not a fast result.

Usage:
    python perf/dsfix/gpu_rfd3_prod.py --arm head-fast --gpu H200 --power-limit 700 \
        --idle-W 80.3 --batches 8 1 --runner /work/v_head/bin/python
"""

import argparse
import gzip
import json
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time

FIN = re.compile(r"(\d\d):(\d\d):(\d\d).*Finished inference batch in ([\d.]+) seconds")
# R4: 9q6y chain A (585 aa) plus the 100-residue designed binder, per the pinned fixture spec.
EXPECT_RESIDUES = 685
HERE = pathlib.Path(__file__).resolve().parent


def sample_power(stop: threading.Event, sink: list) -> None:
    p = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm",
         "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=subprocess.PIPE, text=True)
    try:
        for line in p.stdout:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4:
                try:
                    sink.append((time.time(), float(parts[0]), float(parts[1]),
                                 float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
            if stop.is_set():
                break
    finally:
        p.kill()


def parse_cif(path: pathlib.Path) -> tuple[int, dict, str | None]:
    """Atom count and residues-per-chain from an mmCIF, read off the `_atom_site` loop header.

    Column positions are NOT fixed across writers, so they are resolved by tag name rather than
    index -- a hard-coded slice read the residue name 'ASP' as a coordinate here.
    Returns (n_atoms, {chain: n_residues}, error or None).
    """
    opener = gzip.open if path.suffix == ".gz" else open
    tags: list[str] = []
    n, per_chain = 0, {}
    with opener(path, "rt") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                tags.append(s.split()[0].split(".", 1)[1])
                continue
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if not tags:
                return 0, {}, f"{path.name}: ATOM records before any _atom_site header"
            f = s.split()
            if len(f) != len(tags):
                continue
            row = dict(zip(tags, f))
            for k in ("Cartn_x", "Cartn_y", "Cartn_z"):
                tok = row.get(k)
                if tok is None:
                    return 0, {}, f"{path.name}: no {k} column"
                try:
                    v = float(tok)
                except ValueError:
                    return 0, {}, f"{path.name} has unparseable {k} {tok!r}"
                if v != v or v in (float("inf"), float("-inf")):
                    return 0, {}, f"{path.name} has a non-finite coordinate"
            n += 1
            ch = row.get("auth_asym_id") or row.get("label_asym_id")
            res = row.get("auth_seq_id") or row.get("label_seq_id")
            per_chain.setdefault(ch, set()).add(res)
    return n, {k: len(v) for k, v in sorted(per_chain.items())}, None


def validate(out_dir: pathlib.Path, expect_designs: int, expect_seed: int,
             ref_chains: dict | None) -> tuple[bool, str, dict]:
    """Guard on the output. Returns (ok, why, evidence)."""
    ev: dict = {}
    cifs = sorted(out_dir.glob("*.cif.gz")) + sorted(out_dir.glob("*.cif"))
    ev["n_designs"] = len(cifs)
    if len(cifs) != expect_designs:
        return False, f"wrote {len(cifs)} designs, expected {expect_designs}", ev

    # RFD3 designs a NEW sequence per design, so sidechain atom counts differ between siblings by
    # construction -- 5316 / 5293 in one measured b=8 point. Atom equality is a folding invariant,
    # not a design one. What must be identical is the residue topology: the 585-residue target and
    # the 100-residue designed binder. That is what is compared within a point and across arms.
    atoms, chains = [], None
    for c in cifs:
        n, nres, err = parse_cif(c)
        if err:
            return False, err, ev
        if n == 0:
            return False, f"{c.name} has 0 atoms", ev
        atoms.append(n)
        if chains is None:
            chains = nres
        elif nres != chains:
            return False, f"{c.name} has chains {nres}, sibling designs have {chains}", ev
    ev["residues_per_chain"] = chains
    ev["atoms_min"], ev["atoms_max"] = min(atoms), max(atoms)
    ev["atoms_median"] = round(statistics.median(atoms))
    # RFD3 appends the designed binder to the target's chain rather than giving it a new one, so the
    # output is a single chain A of 685 = 585 target + 100 binder. The residue TOTAL is therefore the
    # check that the design asked for is the design produced.
    ev["residues_total"] = sum((chains or {}).values())
    if ref_chains is not None and chains != ref_chains:
        return False, f"chains {chains} != reference {ref_chains}", ev
    if ev["residues_total"] != EXPECT_RESIDUES:
        return False, (f"{ev['residues_total']} residues, expected {EXPECT_RESIDUES} "
                       f"(585 target + 100 designed binder): {chains}"), ev

    # RFD3 writes the seed at the top level of each design's sidecar JSON, alongside ckpt_path.
    seeds, ckpts = set(), set()
    for j in sorted(out_dir.glob("*.json")):
        try:
            d = json.loads(j.read_text())
        except Exception:
            continue
        if d.get("seed") is not None:
            seeds.add(d["seed"])
        if d.get("ckpt_path"):
            ckpts.add(str(d["ckpt_path"]))
        for m in (d.get("prediction_metadata") or []):
            if isinstance(m, dict) and "seed" in m:
                seeds.add(m["seed"])
    ev["ckpt_paths"] = sorted(ckpts)
    ev["seeds"] = sorted(s for s in seeds if s is not None)
    if ev["seeds"] and ev["seeds"] != [expect_seed]:
        return False, f"metadata seed {ev['seeds']} != requested {expect_seed}", ev
    if not ev["seeds"]:
        return False, "no seed found in output metadata", ev
    return True, "", ev


def run_point(args, b: int) -> dict | None:
    tag = f"{args.arm}_b{b}"
    out_dir = pathlib.Path(args.work) / "out" / tag
    counts = pathlib.Path(args.work) / "out" / f"counts_{tag}.json"
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)

    cmd = [args.runner, str(HERE / "gpu_rfd3_run.py"), "--counts", str(counts), "--",
           "design", f"out_dir={out_dir}", f"inputs={args.inputs}",
           f"inference_sampler.num_timesteps={args.timesteps}",
           f"diffusion_batch_size={b}", f"n_batches={args.n_batches}",
           f"seed={args.seed}", "skip_existing=False"]

    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, cwd=args.work)
    wall = time.time() - t0
    stop.set()
    th.join(timeout=5)
    log = pr.stdout + pr.stderr

    reps, stamps = [], []
    day0 = time.localtime(t0)
    for m in FIN.finditer(log):
        hh, mm, ss, secs = int(m[1]), int(m[2]), int(m[3]), float(m[4])
        stamps.append(time.mktime((day0.tm_year, day0.tm_mon, day0.tm_mday, hh, mm, ss, 0, 0, -1)))
        reps.append(secs)
    if len(reps) < 2:
        print(f"[{tag}] FAILED rc={pr.returncode}\n{log[-4000:]}", flush=True)
        return None

    warm = reps[1:]
    med = statistics.median(warm)
    win = [s for s in samples if stamps[0] <= s[0] <= stamps[-1]]
    ok, why, ev = validate(out_dir, b * args.n_batches, args.seed, args.ref_chains)

    rec = {
        "arm": args.arm, "gpu": args.gpu, "batch": b, "timesteps": args.timesteps,
        "n_batches": args.n_batches, "n_warm": len(warm),
        "batch_s_median": round(med, 4),
        "batch_s_min": round(min(warm), 4), "batch_s_max": round(max(warm), 4),
        "spread_pct": round(100 * (max(warm) - min(warm)) / med, 3),
        "cold_batch_s": round(reps[0], 4), "reps_s": [round(x, 4) for x in reps],
        "s_per_design": round(med / b, 4),
        "designs_per_hour": round(3600 * b / med, 2),
        "process_wall_s": round(wall, 1),
        "power_W_median": round(statistics.median([s[1] for s in win]), 1) if win else None,
        "power_W_min": round(min(s[1] for s in win), 1) if win else None,
        "power_W_max": round(max(s[1] for s in win), 1) if win else None,
        "power_pct_limit": round(100 * statistics.median([s[1] for s in win]) / args.power_limit, 1)
        if win else None,
        "x_idle": round(statistics.median([s[1] for s in win]) / args.idle_W, 2) if win else None,
        "util_pct_median": round(statistics.median([s[2] for s in win]), 1) if win else None,
        "mem_MiB_max": round(max(s[3] for s in win)) if win else None,
        "n_power_samples": len(win),
        "valid": ok, "valid_why": why, "evidence": ev,
        "counts": json.loads(counts.read_text()) if counts.exists() else None,
        "inputs_sha256": args.inputs_sha256, "returncode": pr.returncode,
    }
    with pathlib.Path(args.out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{tag}] batch={med:.2f}s  {rec['s_per_design']:.3f} s/design  "
          f"{rec['designs_per_hour']:.1f} designs/h  {rec['power_W_median']}W  "
          f"util {rec['util_pct_median']}%  valid={ok} {why}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="head-fast | head-sparse | pip-0.2.0 | head-cueq")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--runner", required=True, help="python of the venv holding this arm's install")
    ap.add_argument("--work", default="/work")
    ap.add_argument("--inputs", default="perf/dsfix/fixtures/rfd3_R4_gpu.json")
    ap.add_argument("--inputs-sha256",
                    default="647e066a983e66184e16bf7696b6e731f354e4161c6e764b292e1f9a15c00eef")
    ap.add_argument("--out", default="/work/results/rfd3_prod.jsonl")
    ap.add_argument("--batches", type=int, nargs="+", default=[8])
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=4, help="includes the discarded cold batch")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--power-limit", type=float, required=True)
    ap.add_argument("--idle-W", type=float, required=True)
    ap.add_argument("--ref-chains", type=json.loads, default=None,
                    help='residues-per-chain the designs must match, e.g. \'{"A":585,"B":100}\'. '
                         "Atom counts differ per design by construction; the topology does not.")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-check the designs already on disk and rewrite the JSONL's verdicts, "
                         "without touching the GPU. For when the validator was wrong, not the run.")
    args = ap.parse_args()

    if args.revalidate:
        jl = pathlib.Path(args.out)
        recs = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
        for r in recs:
            od = pathlib.Path(args.work) / "out" / f"{r['arm']}_b{r['batch']}"
            ok, why, ev = validate(od, r["batch"] * r["n_batches"], args.seed, args.ref_chains)
            r["valid"], r["valid_why"], r["evidence"] = ok, why, ev
            print(f"[{od.name}] valid={ok} {why} atoms={ev.get('atoms_min')}-{ev.get('atoms_max')} "
                  f"chains={ev.get('residues_per_chain')} seeds={ev.get('seeds')}", flush=True)
        jl.write_text("".join(json.dumps(r) + "\n" for r in recs))
        sys.exit(0 if all(r["valid"] for r in recs) else 1)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    for b in args.batches:
        rec = run_point(args, b)
        if rec is None or not rec["valid"]:
            bad += 1
    print(f"[{args.arm}] done, {bad} invalid point(s)", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

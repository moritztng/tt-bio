"""Mol-1 on an NVIDIA GPU, timed under state/mol1/PROTOCOL.md.

K resident processes share the one GPU (upstream predicts one job at a time per process, so concurrency is the GPU's
batch knob). Each process builds upstream's runner once; the parent then runs rounds: every process predicts one
seed of the same frozen input through upstream's own `infer_predict`, and a round's wall runs from the start barrier
to the last process's written output. Round 0 is cold (kernel JIT / autotune), rounds 1..warm are warm. Every round
appends one line to `--out/runs.jsonl` as it finishes.

    python mol1_gpu.py --input inputs/mol1-512.json --size 512 --arm fpf-triatt --k 2 --warm 5 --out DIR \
        [--trimul flashpairformer] [--triatt triattention] [--tf32] [--chunk5] [--seed0 101]

The recipe (20 cycles, 200 steps, 5 samples, bf16, MSA from the input's a3m files, no templates) is fixed here and
is NOT settable: an arm that does less work is not an arm.
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

RECIPE = dict(n_cycle=20, n_step=200, n_sample=5, dtype="bf16", use_msa=True, use_template=False)


def worker(rank, a, todo, done):
    try:
        from mol.runner.batch_inference import get_default_runner
        from mol.runner.inference import infer_predict
        t0 = time.time()
        mem = {"infer_setting.sample_diffusion_chunk_size": 5} if a.chunk5 else None
        runner = get_default_runner(seeds=[a.seed0], trimul_kernel=a.trimul, triatt_kernel=a.triatt,
                                    enable_tf32=a.tf32, load_checkpoint_path=a.ckpt, memory_settings=mem, **RECIPE)
        cfg = runner.configs
        cfg.input_json_path = str(a.input)
        done.put(("ready", rank, time.time() - t0, None))
    except Exception:
        done.put(("ready", rank, 0.0, traceback.format_exc()))
        return
    while (job := todo.get()) is not None:
        rnd, seed = job
        d = a.out / f"r{rnd}" / f"p{rank}"
        cfg.dump_dir, cfg.seeds = str(d), [seed]
        runner.dump_dir, runner.error_dir = str(d), str(d / "ERR")
        runner.init_dumper(need_atom_confidence=cfg.need_atom_confidence,
                           sorted_by_ranking_score=cfg.sorted_by_ranking_score)
        err = None
        try:
            infer_predict(runner, cfg)
        except Exception:
            err = traceback.format_exc()[-2000:]
        done.put(("done", rank, time.time(), err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="one frozen job JSON from state/mol1/inputs/")
    ap.add_argument("--size", required=True, type=int)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--k", type=int, default=1, help="concurrent resident processes on the GPU")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed0", type=int, default=101)
    ap.add_argument("--trimul", default="cuequivariance")
    ap.add_argument("--triatt", default="cuequivariance")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--chunk5", action="store_true", help="denoise the 5 samples together")
    ap.add_argument("--ckpt", default=os.environ.get("MOL1_CKPT"))
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = a.out / "runs.jsonl"
    base = {"size": a.size, "arm": a.arm, "batch": a.k, "session": os.environ.get("MOL1_SESSION", ""),
            "sha": os.environ.get("MOL1_SHA", ""), "host": os.environ.get("MOL1_HOST", ""), "raw": str(a.out),
            "knobs": {"trimul": a.trimul, "triatt": a.triatt, "tf32": a.tf32, "chunk5": a.chunk5}}

    def emit(rec):
        rec["t"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        with runs.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    ctx = mp.get_context("spawn")
    done, todos = ctx.Queue(), [ctx.Queue() for _ in range(a.k)]
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(r, a, todos[r], done)) for r in range(a.k)]
    for p in procs:
        p.start()
    ready = [done.get() for _ in procs]
    errs = [e for *_, e in ready if e]
    emit({**base, "kind": "cold", "phase": "load", "predictions": 0, "wall_s": round(time.time() - t0, 3),
          **({"status": "FAILED", "error": errs[0][-2000:]} if errs else {})})
    if errs:
        for q in todos:
            q.put(None)
        sys.exit(1)
    seed = a.seed0
    for rnd in range(a.warm + 1):
        seeds = list(range(seed, seed + a.k)); seed += a.k
        t0 = time.time(); start = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r, q in enumerate(todos):
            q.put((rnd, seeds[r]))
        res = [done.get() for _ in procs]
        wall = round(max(t for _, _, t, _ in res) - t0, 3)
        errs = [e for *_, e in res if e]
        cifs = len(list((a.out / f"r{rnd}").rglob("*.cif")))
        made = cifs // RECIPE["n_sample"]
        rec = {**base, "kind": "cold" if rnd == 0 else "warm", "predictions": made, "wall_s": wall,
               "seed": seeds, "start": start}
        if rnd == 0:
            rec["phase"] = "first"
        if errs or made != a.k:
            rec["status"] = "FAILED"
            rec["error"] = (errs[0] if errs else f"wrote {cifs} CIFs, expected {a.k * RECIPE['n_sample']}")[-2000:]
        emit(rec)
        if errs:
            break
    for q in todos:
        q.put(None)
    for p in procs:
        p.join(timeout=60)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()

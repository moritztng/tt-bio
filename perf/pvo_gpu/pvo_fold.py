"""OpenDDE or Mol-1 on one NVIDIA GPU, timed for PVO (state/pvo/GPU.md, modes in state/pvo/MODES.md).

Both are Protenix-family runners with the same API (`get_default_runner` + `infer_predict`), so one driver serves
both. K resident processes share the GPU (upstream predicts one job at a time per process, so K is the only
same-work concurrency knob). Each process builds upstream's runner once; then, for every input in order, the
parent runs rounds 0..warm where all K processes predict that input at consecutive seeds. Round 0 of an input is
its first-of-shape fold (the process's cold fold on the first input); rounds 1..warm are warm. A round's wall runs
from the start barrier to the last process's written output. Every round appends one line to `--out/runs.jsonl`.

A "load" line records spawn -> runner ready (imports, checkpoint load), so the per-container price (what a Modal
user pays when every fold starts a fresh container) is load + the cold fold, and the warm price is the warm mean.

    python pvo_fold.py --model opendde|mol1 --inputs a.json b.json --arm exact --recipe recipe.json \
        --k 1 --warm 3 --out DIR [--dtype bf16] [--trimul auto] [--triatt auto] [--tf32] [--layernorm torch]
        [--chunk N] [--compile dt|all] [--seed0 101]

The recipe (cycles, steps, samples, MSA, templates) comes from the frozen MODES/INPUTS file and is the same for
every arm of a model: an arm that does less work is not an arm. Only how the work runs is an arm knob.
"""
import argparse, json, multiprocessing as mp, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

# mol1_gpu.py: regional compile of the repeated blocks; "all" dies in Inductor on Mol-1 (KeyError 'op11', torch 2.13).
COMPILE_REGIONS = {"all": ("PairformerBlock", "MSABlock", "DiffusionTransformerBlock"),
                   "dt": ("DiffusionTransformerBlock",)}
RECIPE_KEYS = ("n_cycle", "n_step", "n_sample", "use_msa", "use_template")


def build_runner(a, recipe):
    kw = dict(seeds=[a.seed0], trimul_kernel=a.trimul, triatt_kernel=a.triatt, enable_tf32=a.tf32, dtype=a.dtype,
              load_checkpoint_path=a.ckpt, **{k: recipe[k] for k in RECIPE_KEYS})
    if a.model == "mol1":
        from mol.runner.batch_inference import get_default_runner
        from mol.runner.inference import infer_predict
        if a.chunk:
            kw["memory_settings"] = {"infer_setting.sample_diffusion_chunk_size": a.chunk}
        runner = get_default_runner(**kw)
    else:
        from runner.batch_inference import get_default_runner
        from runner.inference import infer_predict
        runner = get_default_runner(model_name=a.model_name, **kw)
        if a.chunk:  # opendde_v1 defaults to 5 (opendde/config/model_base.py:51); set only when the arm says so
            runner.configs.infer_setting.sample_diffusion_chunk_size = a.chunk
    return runner, infer_predict


def worker(rank, a, recipe, todo, done):
    import torch
    try:
        t0 = time.time()
        runner, infer_predict = build_runner(a, recipe)
        if a.compile:
            if a.trimul != "torch":
                import cuequivariance_ops_torch
                cuequivariance_ops_torch.init_triton_cache()
            for m in runner.model.modules():
                if type(m).__name__ in COMPILE_REGIONS[a.compile]:
                    m.compile()
        cfg = runner.configs
        done.put(("ready", rank, time.time() - t0, None, {}))
    except Exception:
        done.put(("ready", rank, 0.0, traceback.format_exc(), {}))
        return
    while (job := todo.get()) is not None:
        inp, rnd, seed = job
        os.chdir(Path(inp).parent)  # MSA paths in the frozen JSONs are relative to the inputs directory
        d = a.out / Path(inp).stem / f"r{rnd}" / f"p{rank}"
        cfg.input_json_path, cfg.dump_dir, cfg.seeds = inp, str(d), [seed]
        runner.dump_dir, runner.error_dir = str(d), str(d / "ERR")
        runner.init_dumper(need_atom_confidence=cfg.need_atom_confidence,
                           sorted_by_ranking_score=cfg.sorted_by_ranking_score)
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        err = None
        try:
            infer_predict(runner, cfg)
        except Exception:
            err = traceback.format_exc()[-2000:]
        mem = {} if not cuda else {"peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        if cuda:
            torch.cuda.synchronize()
        done.put(("done", rank, time.time(), err, mem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("opendde", "mol1"))
    ap.add_argument("--model_name", default="opendde_v1", help="OpenDDE model name (opendde_v1)")
    ap.add_argument("--inputs", required=True, nargs="+", type=Path, help="frozen job JSONs, timed in this order")
    ap.add_argument("--recipe", required=True, type=Path, help="JSON with " + ", ".join(RECIPE_KEYS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--k", type=int, default=1, help="concurrent resident processes on the GPU")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed0", type=int, default=101)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--trimul", default="auto")
    ap.add_argument("--triatt", default="auto")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--chunk", type=int, help="sample_diffusion_chunk_size; unset = upstream default")
    ap.add_argument("--compile", choices=sorted(COMPILE_REGIONS))
    ap.add_argument("--ckpt", default=os.environ.get("PVO_CKPT", ""))
    a = ap.parse_args()
    recipe = json.loads(a.recipe.read_text())
    missing = [k for k in RECIPE_KEYS if k not in recipe]
    if missing:
        sys.exit(f"recipe {a.recipe} lacks {missing}")
    a.out = a.out.resolve()
    a.inputs = [str(p.resolve()) for p in a.inputs]
    # Both upstreams treat a missing MSA path as "search ColabFold" and carry on, which changes the work and the
    # wall (Mol-1 docs/infer_json_format.md:81). Every path must exist before anything is timed.
    for inp in a.inputs:
        for e in json.loads(Path(inp).read_text())[0]["sequences"]:
            for k in ("pairedMsaPath", "unpairedMsaPath"):
                m = e.get("proteinChain", {}).get(k)
                if recipe["use_msa"] and m and not (Path(inp).parent / m).is_file():
                    sys.exit(f"{inp}: {k} {m} does not exist")
    a.out.mkdir(parents=True, exist_ok=True)
    runs = a.out / "runs.jsonl"
    base = {"model": a.model, "arm": a.arm, "k": a.k, "session": os.environ.get("PVO_SESSION", ""),
            "sha": os.environ.get("PVO_SHA", ""), "host": os.environ.get("PVO_HOST", ""), "raw": str(a.out),
            "recipe": {k: recipe[k] for k in RECIPE_KEYS},
            "knobs": {"dtype": a.dtype, "trimul": a.trimul, "triatt": a.triatt, "tf32": a.tf32,
                      "layernorm": os.environ.get("LAYERNORM_TYPE", "torch"), "chunk": a.chunk,
                      "compile": a.compile}}

    def emit(rec):
        rec["t"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        with runs.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    ctx = mp.get_context("spawn")
    done, todos = ctx.Queue(), [ctx.Queue() for _ in range(a.k)]
    t0 = time.time()
    procs = [ctx.Process(target=worker, args=(r, a, recipe, todos[r], done)) for r in range(a.k)]
    for p in procs:
        p.start()
    ready = [done.get() for _ in procs]
    errs = [e for _, _, _, e, _ in ready if e]
    emit({**base, "kind": "load", "predictions": 0, "wall_s": round(time.time() - t0, 3),
          **({"status": "FAILED", "error": errs[0][-2000:]} if errs else {})})
    seed = a.seed0
    for i, inp in enumerate(a.inputs):
        if errs:
            break
        for rnd in range(a.warm + 1):
            seeds = list(range(seed, seed + a.k)); seed += a.k
            t0 = time.time(); start = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for r, q in enumerate(todos):
                q.put((inp, rnd, seeds[r]))
            res = [done.get() for _ in procs]
            wall = round(max(t for _, _, t, _, _ in res) - t0, 3)
            errs = [e for _, _, _, e, _ in res if e]
            cifs = len(list((a.out / Path(inp).stem / f"r{rnd}").rglob("*.cif")))
            made = cifs // recipe["n_sample"]
            rec = {**base, "input": Path(inp).stem, "kind": "warm" if rnd else ("cold" if i == 0 else "first"),
                   "round": rnd, "predictions": made, "wall_s": wall, "seed": seeds, "start": start,
                   "peak_alloc_gib": max(m.get("peak_alloc_gib", 0) for *_, m in res),
                   "peak_reserved_gib": max(m.get("peak_reserved_gib", 0) for *_, m in res)}
            if errs or made != a.k:
                rec["status"] = "FAILED"
                rec["error"] = (errs[0] if errs else f"wrote {cifs} CIFs, expected {a.k * recipe['n_sample']}")[-2000:]
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

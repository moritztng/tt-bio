"""Mol-1 reference predictions for the MOL1 quality set (state/mol1/PROTOCOL.md section 8), on the chosen B200 arm.

    python mol1_ref.py --inputs DIR --out DIR [--k 4] [--seeds 42,66,101,2024,8888] [--selftest-seeds 7,13,77,1234,9999]

--inputs is mol1-quality's tree (INDEX.json: cases -> job JSON, recipes -> recipe, headline_only). Jobs:
paired cases x GPU seeds -> <out>/paired, paired cases x self-test seeds -> <out>/selftest, headline-only cases x GPU
seeds -> <out>/headline, each in upstream's layout <set>/<case>/seed_<s>/predictions/. K resident processes pull from
one queue. A job whose summary JSON already exists is skipped, so a cut-short session resumes where it stopped.
Jobs are grouped by recipe (a worker rebuilds its runner only when the recipe changes), default first.
Every job appends one line to <out>/jobs.jsonl.
"""
import argparse, json, multiprocessing as mp, os, time, traceback
from pathlib import Path

ARM = dict(trimul_kernel="cuequivariance", triatt_kernel="triattention", enable_tf32=True,
           memory_settings={"infer_setting.sample_diffusion_chunk_size": 5})
BASE = dict(n_cycle=20, n_step=200, n_sample=5, dtype="bf16", use_msa=True, use_template=False, use_rna_msa=False)
RECIPES = {"default": {}, "use_msa_false": {"use_msa": False}, "use_template_true": {"use_template": True},
           "use_rna_msa_true": {"use_rna_msa": True}}


def worker(rank, a, todo, done):
    from mol.runner.batch_inference import get_default_runner
    from mol.runner.inference import infer_predict
    os.chdir(a.inputs)  # MSA paths in the job JSONs are relative to the inputs tree
    runner, have = None, None
    while (job := todo.get()) is not None:
        case, rec, s, seed = job
        t0, err = time.time(), None
        try:
            if rec != have:
                runner = None
                runner = get_default_runner(seeds=[seed], load_checkpoint_path=a.ckpt,
                                            **{**BASE, **RECIPES.get(rec, {}), **ARM})
                have = rec
            cfg = runner.configs
            d = a.out / s
            cfg.input_json_path, cfg.dump_dir, cfg.seeds = str(a.inputs / a.index["cases"][case]), str(d), [seed]
            runner.dump_dir, runner.error_dir = str(d), str(d / "ERR")
            runner.init_dumper(need_atom_confidence=cfg.need_atom_confidence,
                               sorted_by_ranking_score=cfg.sorted_by_ranking_score)
            infer_predict(runner, cfg)
        except Exception:
            err = traceback.format_exc()[-1500:]
        done.put({"case": case, "set": s, "seed": seed, "recipe": rec, "rank": rank, "wall_s": round(time.time() - t0, 2),
                  "status": "FAILED" if err else "ok", **({"error": err} if err else {}), "t": time.time()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reverse", action="store_true",
                    help="run the job list from the end, so a second box fills in from the other side of a running one")
    ap.add_argument("--first", default="",
                    help="comma-separated cases run before everything else, seed-major (each case at the first seed, then the next)")
    ap.add_argument("--seeds", default="42,66,101,2024,8888")
    ap.add_argument("--selftest-seeds", default="7,13,77,1234,9999")
    ap.add_argument("--mol-root", default="/root/moldata", help="MOL_ROOT_DIR: checkpoint/ and common/ reference data")
    ap.add_argument("--ckpt", default=None, help="default <mol-root>/checkpoint/mol-1-base.pt")
    a = ap.parse_args()
    os.environ["MOL_ROOT_DIR"] = a.mol_root  # read by upstream's config at import; spawned workers inherit it
    a.ckpt = a.ckpt or f"{a.mol_root}/checkpoint/mol-1-base.pt"
    a.inputs, a.out = a.inputs.resolve(), a.out.resolve()
    a.index = json.loads((a.inputs / "INDEX.json").read_text())
    seeds, selftest = [int(x) for x in a.seeds.split(",")], [int(x) for x in a.selftest_seeds.split(",")]
    head = set(a.index["headline_only"])
    paired = [c for c in a.index["cases"] if c not in head]
    jobs = ([(c, "paired", x) for c in paired for x in seeds] + [(c, "selftest", x) for c in paired for x in selftest]
            + [(c, "headline", x) for c in sorted(head) for x in seeds])
    order = list(RECIPES)
    jobs = [(c, a.index["recipes"].get(c, "default"), s, x) for c, s, x in jobs]
    jobs.sort(key=lambda j: order.index(j[1]) if j[1] in order else len(order))  # stable: set order kept per recipe
    first, allseeds = [c for c in a.first.split(",") if c], seeds + selftest
    jobs.sort(key=lambda j: (0, allseeds.index(j[3]), first.index(j[0])) if j[0] in first else (1,))
    stem = lambda c: Path(a.index["cases"][c]).stem
    todo_jobs = [j for j in jobs if not list((a.out / j[2] / stem(j[0]) / f"seed_{j[3]}" / "predictions").glob("*summary_confidence*"))]
    todo_jobs = todo_jobs[::-1] if a.reverse else todo_jobs
    print(json.dumps({"jobs": len(jobs), "todo": len(todo_jobs), "k": a.k, "reverse": a.reverse}), flush=True)
    ctx = mp.get_context("spawn")
    todo, done = ctx.Queue(), ctx.Queue()
    for j in todo_jobs:
        todo.put(j)
    for _ in range(a.k):
        todo.put(None)
    procs = [ctx.Process(target=worker, args=(r, a, todo, done)) for r in range(a.k)]
    for p in procs:
        p.start()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "jobs.jsonl").open("a") as log:
        for _ in range(len(todo_jobs)):
            if not any(p.is_alive() for p in procs) and done.empty():
                break  # every worker died; what is on disk is kept and the next session resumes from it
            try:
                log.write(json.dumps(done.get(timeout=1800)) + "\n"); log.flush()
            except Exception:
                continue
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()

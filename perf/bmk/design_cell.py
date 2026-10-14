#!/usr/bin/env python3
"""One tt-bio.com design or affinity cell on one chip: warm seconds per design / prediction.

    python perf/bmk/design_cell.py --model boltzgen --repeat 4 --out OUT/result.json [--fast]
    python perf/bmk/design_cell.py --model rfd3     --repeat 4 --out OUT/result.json [--fast]
    python perf/bmk/design_cell.py --model pxdesign --repeat 4 --out OUT/result.json
    python perf/bmk/design_cell.py --model nesso1   --repeat 4 --out OUT/result.json

Run from the tree under test (PYTHONPATH = that tree) with TT_VISIBLE_DEVICES set to one chip. Every model runs the
code a served request runs, at its shipped defaults, on the page's fixture for that row, one design at a time. The
first rep compiles and is dropped; the next --repeat reps are the cell (warm median).

  boltzgen  `tt-bio design --model boltzgen --steps design` on perf/dsfix/fixtures/bg_R3.yaml (414 target + 100
            binder residues, 514 tokens), 500 sampling steps asserted from the progress stream. One CLI process makes
            repeat+2 designs; one design = the wall between consecutive `batch k/N` stamps (trunk, sampler,
            post-processing, write). The cold design and the one that pays the checkpoint switch are dropped.
  rfd3      `tt_bio.rfd3.design.run_design` on perf/dsfix/fixtures/rfd3_R4.json (9q6y chain A, 585 target residues,
            100-residue binder), 200 timesteps, seed 42, batch 1. Cell = the sampler call (the region the page has
            always published); the whole run_design call is kept as wall_s. Same seed every rep, so the CIF digest
            must not move.
  pxdesign  featurise + `ProtenixDesign.design` + write_design_cifs on perf/pxdesign/targets/laczc_512_nomsa.yaml
            (512 target residues, 80-residue binder, 592 tokens), n_step 400, one backbone per call, checkpoint load
            outside the cell (scripts/release_gate.py:run_pxdesign in a loop). Seeds 0,1,..; the last warm rep repeats
            the cold rep's seed and must reproduce its coordinates.
  nesso1    `tt_bio.nesso1.screen`'s per-input work with the model resident: prepare (parse, RDKit conformer, ESM-2)
            + predict on perf/nesso1/inputs/ladder/aa512 (CDK2 512 aa + one ligand, 532 tokens), 5 recycles, bf16
            trunk, fp32 affinity, 256-token pocket crop, the CLI's seed. A fresh output dir per rep, so no rep reuses
            another's featurisation. Cell = prepare + predict; both halves are kept.

PXDesign and Nesso-1 have no fast mode, so --fast is refused for them rather than timed twice.
The output carries the xmodel_ab shape gapboard harvests (folds[] with tag cold/warmN and fold_s) plus per-model
validation; any validation failure exits 1 after writing the file.
"""
import argparse, hashlib, json, os, re, shutil, statistics, subprocess, sys, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=("boltzgen", "rfd3", "pxdesign", "nesso1"))
ap.add_argument("--fast", action="store_true")
ap.add_argument("--repeat", type=int, default=4, help="warm reps in the cell (one cold rep is added and dropped)")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.fast and a.model in ("pxdesign", "nesso1"):
    sys.exit(f"{a.model} has no fast mode")

TREE = Path.cwd()
sys.path.insert(0, str(TREE))
FIX = {"boltzgen": "perf/dsfix/fixtures/bg_R3.yaml", "rfd3": "perf/dsfix/fixtures/rfd3_R4.json",
       "pxdesign": "perf/pxdesign/targets/laczc_512_nomsa.yaml", "nesso1": "perf/nesso1/inputs/ladder/aa512/cdk2_512.yaml"}


def sha(p, n=64):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:n]


def tree_sha():
    f = TREE / "TREE_SHA"
    return f.read_text().strip() if f.exists() else subprocess.run(
        ["git", "-C", str(TREE), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()


def cif_check(path, want_chains):
    """Chain sizes (residues per chain, as a sorted list) and the count of non-finite coordinates."""
    import gemmi
    st = gemmi.read_structure(str(path))
    sizes = sorted(len(ch) for ch in st[0])
    bad = sum(1 for ch in st[0] for r in ch for at in r
              if not all(abs(v) < 1e6 for v in (at.pos.x, at.pos.y, at.pos.z)))
    err = []
    if sizes != sorted(want_chains):
        err.append(f"{Path(path).name}: chains {sizes} != {sorted(want_chains)}")
    if bad:
        err.append(f"{Path(path).name}: {bad} non-finite coordinates")
    return err


import importlib.metadata as im

res = {"model": a.model, "fast": a.fast, "protocol": "bmk-page-design", "size": None, "samples": 1,
       "repeat": a.repeat, "fixture": FIX[a.model], "fixture_sha256": sha(TREE / FIX[a.model]), "tree": tree_sha(),
       "ttnn": im.version("ttnn"), "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "folds": [], "errors": []}
work = a.out.parent / "work"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)


def rep(i):
    return "cold" if i == 0 else f"warm{i}"


def boltzgen():
    n = a.repeat + 2  # + the cold design + the design that pays the checkpoint switch
    cmd = [sys.executable, "-u", "-m", "tt_bio.main", "design", FIX["boltzgen"], "--model", "boltzgen",
           "--steps", "design", "--num_designs", str(n), "--out_dir", str(work / "bg"), "--debug", "--log"]
    cmd += ["--fast"] if a.fast else []
    res["argv"] = cmd
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=dict(os.environ, PYTHONPATH=str(TREE), PYTHONUNBUFFERED="1"))
    stamps, steps, switched, cur, tail = {}, {}, set(), 0, []
    for line in p.stdout:
        tail = (tail + [line])[-60:]
        if m := re.search(r"batch (\d+)/(\d+)", line):
            cur = int(m.group(1)); stamps[cur] = time.time()
        elif m := re.search(r"diff (\d+)/(\d+)", line):
            steps.setdefault(cur + 1, set()).add(int(m.group(2)))
        elif "Switched checkpoint" in line:
            switched.add(cur + 1)
    if p.wait() or len(stamps) < n + 1:
        res["errors"].append(f"rc={p.returncode}, {len(stamps)} stamps: " + "".join(tail)[-2000:])
        return
    drop = {1: "cold"} | {k: "checkpoint switch" for k in switched if k > 1}
    res["dropped"] = drop
    warm = 0
    for k in range(1, n + 1):
        s = stamps[k] - stamps[k - 1]
        tag = "cold" if k == 1 else ("dropped" if k in drop else f"warm{(warm := warm + 1)}")
        res["folds"].append({"tag": tag, "design": k, "fold_s": round(s, 3), "steps": sorted(steps.get(k, []))})
    if any(sorted(v) != [500] for v in steps.values()) or len(steps) != n:
        res["errors"].append(f"sampling steps per design {dict((k, sorted(v)) for k, v in steps.items())}, want 500")
    cifs = sorted((work / "bg/intermediate_designs").glob("*.cif"))
    if len(cifs) != n:
        res["errors"].append(f"{len(cifs)} designs written, expected {n}")
    for c in cifs:
        res["errors"] += cif_check(c, [100, 414])
    res["cif_sha256"] = [sha(c, 16) for c in cifs]


def rfd3():
    import torch
    from tt_bio.main import ensure_rfd3_weights
    from tt_bio.rfd3 import design as D
    from tt_bio.rfd3.sampler import RFD3Sampler
    walls, sample = [], RFD3Sampler.sample

    def timed(self, *x, **k):
        t = time.perf_counter(); out = sample(self, *x, **k); walls.append(time.perf_counter() - t); return out

    RFD3Sampler.sample = timed
    specs = json.loads((TREE / FIX["rfd3"]).read_text())
    ckpt = str(ensure_rfd3_weights(Path.home() / ".boltz"))  # what `tt-bio design --model rfd3` loads
    for i in range(a.repeat + 1):
        d = work / f"r{i}"
        walls.clear()
        t = time.perf_counter()
        with torch.no_grad():
            D.run_design(specs, d, checkpoint_dir=ckpt, from_pdb=True,
                         num_timesteps=200, seed=42, num_designs=1, batch_size=1, fast=a.fast, verbose=False)
        wall = time.perf_counter() - t
        cifs = sorted(d.glob("*.cif"))
        f = {"tag": rep(i), "fold_s": round(walls[0], 3) if walls else None, "wall_s": round(wall, 3),
             "cif_sha256": sha(cifs[0], 16) if cifs else None}
        res["folds"].append(f)
        print(json.dumps(f), flush=True)
        if len(cifs) != 1 or len(walls) != 1:
            res["errors"].append(f"rep {i}: {len(cifs)} CIFs, {len(walls)} sampler calls")
        for c in cifs:
            res["errors"] += cif_check(c, [100, 585])
    if len({f["cif_sha256"] for f in res["folds"]}) != 1:
        res["errors"].append("seed 42 CIF digest moved between reps")


def pxdesign():
    import torch, yaml
    import tt_bio.tenstorrent as T
    from tt_bio.main import ensure_p300_mesh_descriptor, ensure_pxdesign_weights
    from tt_bio.pxdesign.inputs import design_inputs_from_yaml
    from tt_bio.pxdesign.model import ProtenixDesign
    from tt_bio.pxdesign.write import write_design_cifs
    torch.set_grad_enabled(False)
    # The page fixture names the target by the GPU box's absolute path; point it at the CIF it was cut from
    # (same sha256 on both sides) and change nothing else.
    spec = yaml.safe_load((TREE / FIX["pxdesign"]).read_text())
    cif = TREE / "perf/pxdesign/targets" / Path(spec["target"]["file"]).name
    spec["target"]["file"] = str(cif)
    y = work / "laczc_512_nomsa.yaml"
    y.write_text(yaml.safe_dump(spec))
    res["target_cif_sha256"] = sha(cif)
    ensure_p300_mesh_descriptor()
    t = time.perf_counter()
    model = ProtenixDesign.load_from_checkpoint(str(ensure_pxdesign_weights(Path.home() / ".boltz")))
    res["model_init_s"] = round(time.perf_counter() - t, 3)
    seeds = list(range(a.repeat)) + [0]
    for i, seed in enumerate(seeds):
        t0 = time.perf_counter()
        feats = design_inputs_from_yaml(y)
        feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v) for k, v in feats.items()}
        t1 = time.perf_counter()
        coords = model.design(feats, n_step=400, n_sample=1, seed=seed)
        t2 = time.perf_counter()
        rows = write_design_cifs(coords, feats, work / f"r{i}", stem="laczc_512")
        t3 = time.perf_counter()
        f = {"tag": rep(i), "seed": seed, "fold_s": round(t3 - t0, 3), "feat_s": round(t1 - t0, 3),
             "design_s": round(t2 - t1, 3), "write_s": round(t3 - t2, 3), "n_token": int(feats["restype"].shape[0]),
             "coord_sha16": hashlib.sha256(coords.contiguous().numpy().tobytes()).hexdigest()[:16],
             "cif_sha256": sha(rows[0]["cif"], 16)}
        res["folds"].append(f)
        print(json.dumps(f), flush=True)
        if not bool(torch.isfinite(coords).all()):
            res["errors"].append(f"rep {i}: non-finite coordinates")
        if f["n_token"] != 592:
            res["errors"].append(f"rep {i}: {f['n_token']} tokens, want 592")
        res["errors"] += cif_check(rows[0]["cif"], [80])
    if res["folds"][0]["coord_sha16"] != res["folds"][-1]["coord_sha16"]:
        res["errors"].append("seed 0 did not reproduce its cold-rep coordinates")
    T.cleanup()


def nesso1():
    import math, torch
    from tt_bio.main import ensure_p300_mesh_descriptor
    from tt_bio.nesso1 import DEFAULT_SEED, REPORTED_SCALARS, Nesso1
    import tt_bio.nesso1_input as NI
    import tt_bio.weights as W
    from tt_bio._vendor.nesso.data import esm as E
    from tt_bio.nesso1_input import CLI_PREDICT_ARGS, collate, prepare
    from tt_bio.runtime import reuse_freed_heap
    # A served request keeps each target's ESM-2 embedding under the user's cache, so a warm rep would skip the ESM
    # pass a first-seen target pays (and the GPU invocation pays every time). Give run_esm an empty store per rep and
    # load the 650M once, outside the cell, as the trunk weights are.
    esm_once, run_esm, cache_root = {}, NI.run_esm, W.cache_root
    setup = E.setup_esm_model

    def setup_once(*x, **k):
        if "m" not in esm_once:
            esm_once["m"] = setup(*x, **k)
        return esm_once["m"]

    E.setup_esm_model = setup_once

    def fresh_esm(*x, **k):
        W.cache_root = lambda: store
        try:
            return run_esm(*x, **k)
        finally:
            W.cache_root = cache_root

    NI.run_esm = fresh_esm
    reuse_freed_heap()
    ensure_p300_mesh_descriptor()
    t = time.perf_counter()
    model = Nesso1.from_pretrained(use_tenstorrent=True, trunk_fp32=False, affinity_fp32=True)
    model.use_kernels = False
    model.predict_args.update(CLI_PREDICT_ARGS)
    model.predict_args["recycling_steps"] = 5
    model.predict_args["refine_protein_tokens_budget"] = 256
    res["model_init_s"] = round(time.perf_counter() - t, 3)
    src = (TREE / FIX["nesso1"]).parent
    for i in range(a.repeat + 1):
        store = work / f"esm{i}"
        t0 = time.perf_counter()
        dataset, manifest, failed = prepare(src, work / f"r{i}", num_workers=0)
        torch.manual_seed(DEFAULT_SEED)  # where screen() seeds: the featuriser's random roto-translation
        feats = collate(dataset[0])
        t1 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(feats)
        t2 = time.perf_counter()
        sc = {k: float(pred[k].reshape(-1)[0]) for k in REPORTED_SCALARS if k in pred}
        f = {"tag": rep(i), "fold_s": round(t2 - t0, 3), "prepare_s": round(t1 - t0, 3),
             "predict_s": round(t2 - t1, 3), "n_tokens": int(feats["token_pad_mask"].shape[-1]), "scalars": sc,
             "cif_sha256": hashlib.sha256(json.dumps(sc, sort_keys=True).encode()).hexdigest()[:16]}
        res["folds"].append(f)
        print(json.dumps(f), flush=True)
        f["esm_computed"] = len(list(store.rglob("*.safetensors")))
        if failed or len(manifest.records) != 1 or not f["esm_computed"]:
            res["errors"].append(f"rep {i}: {len(manifest.records)} records, failed {failed}")
        if not sc or not all(math.isfinite(v) for v in sc.values()):
            res["errors"].append(f"rep {i}: scalars {sc}")
    if len({f["cif_sha256"] for f in res["folds"]}) != 1:
        res["errors"].append("seeded scalars moved between reps")


try:
    {"boltzgen": boltzgen, "rfd3": rfd3, "pxdesign": pxdesign, "nesso1": nesso1}[a.model]()
except Exception as e:
    import traceback
    res["errors"].append(f"{type(e).__name__}: {e}")
    traceback.print_exc()
warm = sorted(f["fold_s"] for f in res["folds"] if f["tag"].startswith("warm") and f["fold_s"] is not None)
if len(warm) < a.repeat:
    res["errors"].append(f"{len(warm)} warm reps, want {a.repeat}")
if warm:
    res["warm_median_s"] = round(statistics.median(warm), 3)
    res["warm_spread_pct"] = round(100 * (warm[-1] - warm[0]) / statistics.median(warm), 2)
res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
res["ok"] = not res["errors"]
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
print(f"[{a.model}{' fast' if a.fast else ''}] warm median {res.get('warm_median_s')} s over n={len(warm)}, "
      f"spread {res.get('warm_spread_pct')} %, ok={res['ok']} {res['errors'][:3]}", flush=True)
sys.exit(0 if res["ok"] else 1)

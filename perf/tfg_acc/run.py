"""Fold one TFG panel target on one Tenstorrent chip the way upstream's reference run does, for the accuracy row.

Per seed and condition this is upstream's `opendde pred --seeds S --sample 5 --cycle 10 --step 200` with the
ABAG checkpoint: `unconstrained` runs the standard sampler (TFG off), `contact` and `pocket` run with
--use_tfg_guidance. The input is upstream's JSON (perf/tfg_ref/build_inputs.py) with its precomputed MSAs, read
cache-only, so TT and GPU fold the same alignment. The model is loaded once per target and every fold goes
through the serving worker's `predict_one`, the code JapanFold runs. A trunk cache shared by all seeds and
conditions computes the trunk once per target (the trunk does not depend on the seed or the constraint).

Output, in the layout perf/tfg_ref/score.py reads (so TT and GPU are scored by one scorer):
    OUT/<cond>/seed_<s>/<tid>_<cond>/seed_<s>/predictions/<tid>_<cond>_sample_<r>.cif
    OUT/<cond>/seed_<s>/<tid>_<cond>/seed_<s>/predictions/<tid>_<cond>_summary_confidence_sample_<r>.json
r is tt-bio's confidence rank (0 = best); the JSON carries ranking_score. OUT/runs.jsonl gets one record per
fold (wall time, chip, AICLK during the fold, sha). A finished (cond, seed) is skipped on rerun.

    python perf/tfg_acc/run.py --panel ~/tfg-ref/panel --target 1bzq --out RUN/1bzq --chip 3
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--panel", required=True, type=Path, help="dir with <tid>/<tid>_<cond>.json (MSA paths relative to it)")
ap.add_argument("--target", required=True)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--chip", type=int, default=None, help="UMD index for TT_VISIBLE_DEVICES (unset: as the env says)")
ap.add_argument("--seeds", default="101,102,103,104,105")
ap.add_argument("--conds", default="unconstrained,contact,pocket")
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--recycles", type=int, default=10)
ap.add_argument("--model", default="opendde-abag")
ap.add_argument("--share", type=int, default=None, help="host thread share (default: one per chip on the host)")
ap.add_argument("--host-threads", type=int, default=2, help="torch threads for the host guidance work")
ap.add_argument("--dry", action="store_true", help="write the inputs, build the run config and validate the "
                "constraints against the features, then stop before weights and device")
a = ap.parse_args()
a.panel, a.out = a.panel.expanduser().resolve(), a.out.expanduser().resolve()
SEEDS = [int(s) for s in a.seeds.split(",")]
CONDS = a.conds.split(",")
REPO = Path(__file__).resolve().parents[2]
work = a.out / "work"
work.mkdir(parents=True, exist_ok=True)
LOG = (a.out / "runs.jsonl").open("a")


def log(**kw):
    kw["t_unix"] = time.time()
    LOG.write(json.dumps(kw, default=str) + "\n"); LOG.flush(); print(json.dumps(kw, default=str), flush=True)


# ---- input: upstream JSON -> tt-bio YAML + a cache-only MSA dir holding upstream's alignments -----------------
import yaml  # noqa: E402

from tt_bio.cache import paired_msa_dir, seq_hash  # noqa: E402

msa_dir = work / "msa"
msa_dir.mkdir(exist_ok=True)
inputs = {}
for cond in CONDS:
    job = json.loads((a.panel / a.target / f"{a.target}_{cond}.json").read_text())[0]
    seqs, paired = [], {}
    used = {c for ent in job["sequences"] for item in ent.values() for c in item.get("id", [])}
    free = (c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used)
    for ent in job["sequences"]:
        (kind, item), = ent.items()
        if kind == "ligand":  # upstream labels ligand copies after the polymers, in input order
            ids = [next(free) for _ in range(item.get("count", 1))]
            seqs.append({"ligand": {"id": ids if len(ids) > 1 else ids[0], "ccd": item["ligand"].removeprefix("CCD_")}})
            continue
        if kind != "proteinChain":
            sys.exit(f"{a.target}: {kind} entities are not handled by this harness")
        seqs.append({"protein": {"id": item["id"] if len(item["id"]) > 1 else item["id"][0],
                                 "sequence": item["sequence"]}})
        h = seq_hash(item["sequence"])
        if "unpairedMsaPath" in item:
            dst = msa_dir / f"{h}.a3m"
            if not dst.exists():
                shutil.copyfile(a.panel / item["unpairedMsaPath"], dst)
        if "pairedMsaPath" in item:
            paired[item["sequence"]] = a.panel / item["pairedMsaPath"]
    pdir = paired_msa_dir(msa_dir, paired)
    if pdir is not None:
        pdir.mkdir(parents=True, exist_ok=True)
        for s, src in paired.items():
            if not (pdir / f"{seq_hash(s)}.a3m").exists():
                shutil.copyfile(src, pdir / f"{seq_hash(s)}.a3m")
    doc = {"version": 1, "sequences": seqs}
    ids = [list(v["id"]) if isinstance(v["id"], list) else [v["id"]] for e in seqs for v in e.values()]
    bonds = []
    for b in job.get("covalent_bonds", []):  # no copy given: copy k bonds to copy k (equal counts)
        c1, c2 = ids[int(b["entity1"]) - 1], ids[int(b["entity2"]) - 1]
        pairs = ([(c1[int(b["copy1"]) - 1], c2[int(b["copy2"]) - 1])] if "copy1" in b else list(zip(c1, c2)))
        bonds += [{"bond": {"atom1": [x, int(b["position1"]), b["atom1"]], "atom2": [y, int(b["position2"]), b["atom2"]]}}
                  for x, y in pairs]
    if bonds:
        doc["constraints"] = bonds
    if job.get("constraint"):
        doc["constraint"] = job["constraint"]
    y = work / f"{a.target}_{cond}.yaml"
    y.write_text(yaml.safe_dump(doc, sort_keys=False))
    inputs[cond] = y

# ---- the run config exactly as `tt-bio predict` builds it, then the serving worker in-process ----------------
from tt_bio import runtime  # noqa: E402

if a.chip is not None:
    os.environ["TT_VISIBLE_DEVICES"] = str(a.chip)
share = a.share if a.share is not None else len(glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
if share:
    os.environ.update(runtime.host_thread_cap_env(share, None))

from tt_bio import main as M  # noqa: E402

captured = {}


class _Stop(Exception):
    pass


def _grab(*args, **kw):
    captured["payload"] = args[1] if isinstance(args[0], str) else args[0]
    raise _Stop


M._dispatch_run = _grab; M._dispatch_to_controller = _grab
argv = ["predict", str(inputs[CONDS[0]]), "--model", a.model, "--diffusion_samples", str(a.samples),
        "--sampling_steps", str(a.steps), "--recycling_steps", str(a.recycles), "--accelerator", "tenstorrent",
        "--output_format", "cif", "--msa_dir", str(msa_dir), "--msa_cache_only", "--use_tfg_guidance",
        "--trunk_cache", str(work / "trunk"), "--out_dir", str(work / "cli")]
try:
    M.cli.main(argv, standalone_mode=False)
except _Stop:
    pass
cfg0 = dict(captured["payload"]["config"])
if a.dry:
    from tt_bio.main import _read_bio_bonds, _read_bio_chains
    from tt_bio.protenix_data import build_complex_features

    for cond, y in inputs.items():
        chains = _read_bio_chains(y)
        bonds = _read_bio_bonds(y, chains)
        W_specs = __import__("tt_bio.worker", fromlist=["x"])
        cs = W_specs._build_chain_specs(chains, msa_dir, cfg0, protein_only=False)
        pa = W_specs._paired_a3ms(y, chains, msa_dir, cfg0)
        feats = build_complex_features(cs, chain_ids=[c[0] for c in chains], bonds=bonds, paired_a3ms=pa,
                                       modifications=[c[4] for c in chains])
        g = W_specs._tfg_guidance(y, dict(cfg0, use_tfg_guidance=cond != "unconstrained"), feats, chains, bonds)
        log(ev="dry", target=a.target, cond=cond, chains=[c[0] for c in chains], bonds=len(bonds),
            tokens=int(feats["restype"].shape[0]), msa_depth=int(feats["msa"].shape[0]), paired=pa is not None,
            guided=g is not None)
    os._exit(0)

import torch  # noqa: E402

from tt_bio import worker as W  # noqa: E402

W._ensure_local_artifacts(cfg0)
from tt_bio.host_controller import worker_payload  # noqa: E402

winfo = worker_payload(runtime.build_local_workers("tenstorrent", [object()], [a.chip or 0])[0])
W._apply_tt_environment(winfo); W._bind_host_threads()
torch.set_num_threads(a.host_threads)  # guidance is host work; few threads beat many on a loaded box

NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
clk = []


def _sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                if 100 <= v <= 3000:
                    row[n] = v
            except Exception:
                pass
        clk.append((time.monotonic(), row)); time.sleep(1.0)


threading.Thread(target=_sampler, daemon=True).start()


def opened_nodes():
    out = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            out.add(int(t.rsplit("/", 1)[1]))
    return sorted(out)


def aiclk(t0, t1, nodes):
    v = sorted(r[n] for ts, r in clk if t0 <= ts <= t1 for n in nodes if n in r)
    return dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v)) if v else None


sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
if not sha and (REPO / "SHA").exists():
    sha = (REPO / "SHA").read_text().strip()
state = W._WorkerState("tenstorrent")
t = time.monotonic()
state.load_model(cfg0)
state.bind_run("tfg-accuracy", cfg0)
log(ev="load", target=a.target, s=round(time.monotonic() - t, 1), sha=sha, chip=a.chip, host=os.uname().nodename,
    cfg={k: cfg0.get(k) for k in ("model", "sampling_steps", "recycling_steps", "diffusion_samples", "fast")})

for seed in SEEDS:
    for cond in CONDS:
        name = f"{a.target}_{cond}"
        pred = a.out / cond / f"seed_{seed}" / name / f"seed_{seed}" / "predictions"
        if (pred / ".done").exists():
            continue
        sdir = work / "struct" / f"{cond}_s{seed}"
        shutil.rmtree(sdir, ignore_errors=True); sdir.mkdir(parents=True)
        rcfg = dict(cfg0, seed=seed, struct_dir=str(sdir), use_tfg_guidance=cond != "unconstrained")
        t0 = time.monotonic()
        try:
            metrics, _, _ = state.predict_one(inputs[cond], rcfg)
        except Exception as e:  # one failed fold must not lose the target's other folds
            import traceback
            log(ev="fail", target=a.target, cond=cond, seed=seed, err=traceback.format_exc()[-3000:])
            continue
        t1 = time.monotonic()
        runs = metrics.get("all_runs") or [{"rank": 0, **metrics}]
        pred.mkdir(parents=True, exist_ok=True)
        for r in runs:
            k = r["rank"]
            src = sdir / (f"{name}.cif" if k == 0 else f"{name}_model_{k}.cif")
            shutil.copyfile(src, pred / f"{name}_sample_{k}.cif")
            (pred / f"{name}_summary_confidence_sample_{k}.json").write_text(json.dumps(
                {"ranking_score": r["confidence_score"], "ptm": r["ptm"], "iptm": r["iptm"], "plddt": r["plddt"]}))
        (pred / ".done").touch()
        log(ev="fold", target=a.target, cond=cond, seed=seed, wall_s=round(t1 - t0, 1), tokens=metrics.get("n_tokens"),
            aiclk=aiclk(t0, t1, opened_nodes()), sha=sha, chip=a.chip,
            scores=[round(r["confidence_score"], 4) for r in runs])
log(ev="end", target=a.target)
os._exit(0)

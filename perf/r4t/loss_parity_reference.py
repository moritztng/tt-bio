"""G3 loss-parity reference: upstream RFD4-Proteina's own forward + loss on fixed batches, on CPU.

For each batch this runs exactly the forward half of `Proteina.training_step` (add_clean_samples,
shared-per-example t, corrupt_batch, call_nn at a fixed recycle depth, compute_loss, the
exploration-k winner, optimization_terms) with the released weights, and records everything a
second implementation needs to replay it: the network's input batch after corruption, the torch
RNG state the network call starts from, the recycle depth, the network outputs, every loss term
per physical row, the winners and the total. The TT side reads this file, runs its own forward on
the recorded batch and grades with `tt_bio.train.parity.loss_parity` (PLAN.md G3: |rel| <= 1e-2
bf16, <= 1e-4 fp32, n >= 8, D14: reference = upstream on CPU).

Data is upstream's own pipeline over its published test_dataset, the same settings as the
committed fixture (`perf/r4t/build_fixture.py`), one example x exploration_k 4 per batch at I=256.

    RFD4_REPO=~/rfd4-upstream/RFD4-Proteina DATA_PATH=~/r4t/objective/data HF_HOME=~/r4t/gpu-baseline/hf \\
      ~/rfd4-cpu-venv/bin/python perf/r4t/loss_parity_reference.py --n 8 --out ~/r4t/accuracy/ref
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CROP, K = 256, 4
HF_REPO, HF_REV = "nvidia/RFD4-Proteina", "a06e2509c7033b010fc43614e9fd840303588502"


def flat(d, prefix=""):
    """Nested dict of tensors -> {'a/b': ndarray}; non-tensor leaves are skipped."""
    import torch
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= flat(v, key + "/")
        elif isinstance(v, torch.Tensor):
            out[key] = v.detach().cpu().numpy()
    return out


def load_batch(path, dtype=None):
    """A dumped `batch<NN>.npz` back into the dict `call_nn` receives, with `in/a/b` unflattened.

    `strict_feats` is set: the feature factory fills a MISSING feature with a silent zero default
    unless it is on (`nn/feature_factory.py:196`), and a replay that quietly lost a key must raise
    rather than train against zeros (LESSONS-OF3T T1). It is not in the dump because it is a flag,
    not data; neither is `example_id_batch`, which only names the example in an error message.
    """
    import torch
    z = np.load(path, allow_pickle=False)
    batch = {}
    for key in z.files:
        if not key.startswith("in/"):
            continue
        parts = key[3:].split("/")
        v = torch.as_tensor(z[key])
        if dtype is not None and v.is_floating_point():
            v = v.to(dtype)
        d = batch
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    batch["strict_feats"] = True
    meta = {"depth": int(z["depth"]), "seed": int(z["seed"]), "winners": z["winners"],
            "rng_state": torch.as_tensor(z["rng_state"], dtype=torch.uint8)}
    recorded = {k[4:]: z[k] for k in z.files if k.startswith("out/")}
    losses = {k[5:]: z[k] for k in z.files if k.startswith("loss/")}
    return batch, meta, recorded, losses


def replay(repo, out_dir, dtype):
    """Prove a dump is sufficient: rerun `call_nn` from the file alone and match the recorded outputs.

    This is the artifact's contract check. If the dump is missing something the network reads, the
    replayed outputs move (or the strict feature factory raises) and this fails -- which is what a
    second implementation grading itself against these files needs to be able to rely on.
    """
    import torch
    _, model = build_model(repo, dtype)
    worst, rows = 0.0, []
    for path in sorted(Path(out_dir).glob("batch*.npz")):
        batch, meta, recorded, _ = load_batch(path, dtype)
        torch.set_rng_state(meta["rng_state"])
        with torch.no_grad():
            out = flat(model.call_nn(batch, n_recycle=meta["depth"]))
        miss = sorted(set(recorded) - set(out)) + sorted(set(out) - set(recorded))
        rel = {k: float(np.max(np.abs(out[k] - recorded[k])) /
                        max(float(np.max(np.abs(recorded[k]))), 1e-30)) for k in recorded if k in out}
        bad = max(rel.values()) if rel else float("inf")
        worst = max(worst, bad)
        rows.append({"file": path.name, "worst_rel": bad, "missing_or_extra": miss,
                     "per_tensor": rel})
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"batches": len(rows), "worst_rel": worst,
                      "verdict": "REPLAYABLE" if rows and worst == 0.0 else "NOT BIT-EXACT"}))
    return 0 if rows and worst == 0.0 else 1


def emit_ref(out_dir: Path, dest: Path) -> int:
    """summary.json -> the `ref` half of `perf/r4t/grade_parity.py --parity`'s document.

    The TT side fills `ours` with the same term names on the same batches and the verdict is
    mechanical. `total` is carried as a term because the grader compares it like any other: a
    loss that drops a term matches on everything it still computes.
    """
    s = json.loads((Path(out_dir) / "summary.json").read_text())
    ref = [dict(b["terms"], total=b["total"]) for b in s["batches"]]
    doc = {"dtype": "float32", "ref": ref, "ours": [],
           "provenance": {"reference": "upstream RFD4-Proteina on CPU (D14)",
                          "upstream_sha": s.get("upstream_sha"), "hf": s.get("hf"),
                          "crop": s.get("crop"), "k": s.get("k"),
                          "depth": sorted({b["depth"] for b in s["batches"]}),
                          "dtype": sorted({b["dtype"] for b in s["batches"]}),
                          "batches": [{"i": b["batch"], "example_id": b["example_id"],
                                       "tokens": b["tokens"], "seed": b["seed"],
                                       "zero_terms": b.get("zero_terms", [])} for b in s["batches"]],
                          "dumps": str(Path(out_dir).resolve())}}
    dest.write_text(json.dumps(doc, indent=1))
    print(f"{dest}: {len(ref)} reference batches, terms {sorted(ref[0]) if ref else []}")
    return 0


def build_model(repo: Path, dtype):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    sys.path.insert(0, str(repo / "src"))
    from rfproteina.proteina import Proteina
    from rfproteina.train import _adopt_bundle_architecture
    from rfproteina.utils.hub import load_pretrained_bundle

    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="training_proteina",
                      overrides=["experiment=tutorial/finetune_test_dataset", "+single=true", "+nolog=true"])
    payload, state = load_pretrained_bundle(HF_REPO, revision=HF_REV)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "nn.use_torch_compile", False, force_add=True)
    _adopt_bundle_architecture(cfg, OmegaConf.create(payload["config"]))
    OmegaConf.update(cfg, "nn.use_torch_compile", False, force_add=True)
    model = Proteina(cfg)
    model.load_state_dict(state, strict=True)
    model = model.to(dtype)
    model.train()   # training_step's mode; the eval-only branches (cudagraphs) stay off on CPU
    return cfg, model


def batches(cfg, n, seed, min_tokens, start=0):
    """Upstream's pipeline, skipping examples too small to grade a relative bar.

    The first system in upstream's published test_dataset crops to 4 real tokens, and at that
    size four of the five optimization terms come out identically zero (no chiral centre, no
    unindexed token, no conditioned residue), so a relative tolerance has nothing to divide by.
    A zero term is not a parity pass, so such a batch is skipped rather than graded.
    """
    import torch
    from hydra.utils import instantiate
    from rfproteina.datapipes.collate import collate_batch

    ds = instantiate(cfg.datasets.train.test_dataset.dataset)
    torch.manual_seed(seed)
    np.random.seed(seed)
    got = 0
    for i in range(start, len(ds)):
        if got == n:
            return
        try:
            s = ds[i]
        except Exception as e:                       # upstream's own pipeline rejects systems
            print(f"skip {i}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            continue
        b = collate_batch([s], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
                          fixed_dimensions={"I": CROP}, keep_atom_array=False, num_replicates=K)
        feats = b["feats"] if "feats" in b else b
        tokens = int(feats["padding_mask"][0].sum())
        if tokens < min_tokens:
            print(f"skip {i} ({s['example_id']}): {tokens} tokens < {min_tokens}", file=sys.stderr)
            continue
        got += 1
        yield str(s["example_id"]), feats
    raise SystemExit(f"only {got} of {n} examples were >= {min_tokens} tokens and survived the pipeline")


def run(model, batch, seed, depth, dtype):
    """`Proteina.training_step`'s forward half, at a recycle depth passed in."""
    import torch
    from rfproteina.utils.torch import minimum_loss_attempt_indices

    fm = model.fm
    torch.manual_seed(seed)
    batch = {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in batch.items()}
    batch["coords_nm"] = batch["coords"] / 10.0
    batch["coord_mask"] = batch["atom37_mask"]
    with torch.no_grad():
        batch = model.add_clean_samples(batch)
        t = fm.sample_t(shape=(batch["padding_mask"].shape[0] // K,), device=batch["padding_mask"].device)
        batch = fm.corrupt_batch(batch, t={dm: v.repeat_interleave(K) for dm, v in t.items()})
        nn_in = flat(batch)
        rng = torch.get_rng_state()
        t0 = time.time()
        nn_out = model.call_nn(batch, n_recycle=depth)
        wall = time.time() - t0
        losses = fm.compute_loss(batch=batch, nn_out=nn_out)
        names = model.cfg_exp.training.get("exploration_losses", fm.data_modes)
        win = minimum_loss_attempt_indices(losses=losses, loss_names=list(names), attempts_per_example=K)
        sel = {k: v[win] for k, v in losses.items()}
        terms = {k: float(v.mean()) for k, v in fm.optimization_terms(sel).items()}
    return nn_in, rng.numpy(), flat(nn_out), {k: v.detach().numpy() for k, v in losses.items()}, \
        win.numpy(), terms, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=145, help="data pipeline seed (hero_pretrain's)")
    ap.add_argument("--depth", type=int, default=2,
                    help="recycle depth; >= 2 so the cycle projections are in the graded forward")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    ap.add_argument("--min-tokens", type=int, default=32,
                    help="skip examples smaller than this; see batches() on the 4-token system")
    ap.add_argument("--start", type=int, default=0, help="first dataset index to consider")
    ap.add_argument("--index-offset", type=int, default=0,
                    help="number the dumps from here, to extend an existing set without overwriting")
    ap.add_argument("--emit-ref", type=Path,
                    help="write the grader's `ref` half from --out/summary.json and exit")
    ap.add_argument("--replay", action="store_true",
                    help="do not generate: rerun call_nn from the dumps in --out and match them")
    a = ap.parse_args()

    if a.emit_ref:                     # needs numpy and json only, not upstream's stack
        sys.exit(emit_ref(a.out, a.emit_ref))

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    dtype = getattr(torch, a.dtype)
    repo = Path(os.environ["RFD4_REPO"]).expanduser()
    if a.replay:
        sys.exit(replay(repo, a.out, dtype))
    cfg, model = build_model(repo, dtype)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for j, (eid, batch) in enumerate(batches(cfg, a.n, a.seed, a.min_tokens, a.start)):
        i = j + a.index_offset
        seed = 0x4733_0000 + i
        nn_in, rng, out, losses, win, terms, wall = run(model, batch, seed, a.depth, dtype)
        np.savez_compressed(a.out / f"batch{i:02d}.npz", **{f"in/{k}": v for k, v in nn_in.items()},
                            **{f"out/{k}": v for k, v in out.items()},
                            **{f"loss/{k}": v for k, v in losses.items()},
                            rng_state=rng, winners=win, depth=np.int64(a.depth), seed=np.int64(seed))
        row = {"batch": i, "example_id": eid, "seed": seed, "depth": a.depth, "dtype": a.dtype,
               "tokens": int(batch["padding_mask"][0].sum()), "winners": win.tolist(),
               "terms": terms, "total": sum(terms.values()), "nn_wall_s": round(wall, 1),
               "zero_terms": sorted(k for k, v in terms.items() if v == 0.0)}
        summary.append(row)
        print(json.dumps(row), flush=True)
        # A term that is zero in EVERY graded batch is graded trivially (the bar passes both
        # sides at zero) but never exercised, so coverage is reported beside the values. These
        # PLINDER systems carry no chiral centre, unindexed token or conditioned residue each,
        # so covering all five terms means choosing examples that between them do.
        covered = {k for row in summary for k, v in row["terms"].items() if v != 0.0}
        (a.out / "summary.json").write_text(json.dumps(
            {"upstream_sha": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(),
             "hf": f"{HF_REPO}@{HF_REV}", "crop": CROP, "k": K, "depth": a.depth,
             "term_coverage": {k: sum(1 for r in summary if r["terms"].get(k, 0.0) != 0.0)
                               for k in (summary[0]["terms"] if summary else {})},
             "terms_never_exercised": sorted(set(summary[0]["terms"]) - covered) if summary else [],
             "batches": summary}, indent=1))


if __name__ == "__main__":
    main()

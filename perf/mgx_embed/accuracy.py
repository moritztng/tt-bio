"""Embedding accuracy at length: the shipped device path against each model's own fp32 upstream.

    python3 perf/mgx_embed/accuracy.py --model esmc-6b --pdb mtor.pdb --lengths 76,1536,2000 \
        --out rows.jsonl [--fast] [--foldseek BIN]

One real protein for every length: the first L residues of the --pdb chain (the AlphaFold model
of human mTOR, 2549 aa, is what the MGX row used). A tiled sequence repeats every few hundred
residues and attention finds the repeat, so it is an easier input than any real one. The 76 aa
prefix is the in-run control, so a long-length number sits beside a short one scored the same way.

References, all fp32 on CPU and built from each model's own upstream code:
  esmc-300m/600m  the esm-repo ESMC (tests/esmc_reference.py) on the released .pth
  esmc-6b         the same class at the 6B config on the released safetensors
  saprot-*        HuggingFace EsmForMaskedLM on the released checkpoint
For saprot with --foldseek, the 3Di tokens come from foldseek on the same structure, so the
structure-token axis is exercised; without it every 3Di token is '#' (sequence-only mode).

Flattened PCC is dominated by a few large channels and reads 0.999 on vectors that disagree
residue by residue, so each row also carries the per-residue cosine (min, 1st percentile,
median), the pooled cosine and the relative L2 error.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "tests"):
    sys.path.insert(0, str(p))
from perf.clocksample import during  # noqa: E402

THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def pdb_sequence(path):
    seen, out = set(), []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            key = (line[21], line[22:27])
            if key not in seen:
                seen.add(key)
                out.append(THREE.get(line[17:20], "X"))
    return "".join(out)


def metrics(dev, ref):
    dev, ref = dev.astype(np.float64), ref.astype(np.float64)
    cos = (dev * ref).sum(1) / (np.linalg.norm(dev, axis=1) * np.linalg.norm(ref, axis=1))
    pd, pr = dev.mean(0), ref.mean(0)
    return {"pcc": float(np.corrcoef(dev.ravel(), ref.ravel())[0, 1]),
            "cos_min": float(cos.min()), "cos_p01": float(np.percentile(cos, 1)),
            "cos_median": float(np.median(cos)),
            "pooled_cos": float(pd @ pr / (np.linalg.norm(pd) * np.linalg.norm(pr))),
            "rel_l2": float(np.linalg.norm(dev - ref) / np.linalg.norm(ref)),
            "finite": bool(np.isfinite(dev).all())}


def esmc_reference(model):
    from tt_bio import esmc
    if model == "esmc-6b":
        from esmc6b_embed_parity import _build_reference
        ref = _build_reference()
    else:
        from huggingface_hub import hf_hub_download
        from esmc_embed_parity import load_reference
        _cfg, repo, wpath = esmc.CONFIGS[model]
        sd = torch.load(hf_hub_download(repo, wpath), map_location="cpu", weights_only=False)
        ref = load_reference(model, sd.get("state_dict", sd))
    return lambda aa, _s: ref(esmc.tokenize(aa))[1][0][1:-1].float().numpy()


def saprot_reference(model):
    from transformers import EsmForMaskedLM
    from tt_bio import saprot, weights
    # The directory the device loader reads, so both sides hold the same checkpoint.
    ref = EsmForMaskedLM.from_pretrained(weights.fetch(model), dtype=torch.float32).eval()
    # Same fused token ids the device sees, so the comparison is the encoder and not a tokenizer.
    return lambda aa, s: ref.esm(saprot.tokenize(aa, s)).last_hidden_state[0][1:-1].float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--lengths", default="76,1536,2000")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--foldseek", default=None, help="saprot only: take 3Di from the structure")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-cache", default=None,
                    help="directory of cached reference arrays, keyed by model/L/3Di")
    ap.add_argument("--ref-only", action="store_true", help="fill --ref-cache, open no device")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    is_saprot = a.model.startswith("saprot")

    full = pdb_sequence(a.pdb)
    struc = ""
    if is_saprot and a.foldseek:
        from tt_bio import saprot
        aa, s3 = next(iter(saprot.foldseek_3di(a.pdb, a.foldseek).values()))
        assert aa == full, "foldseek read a different sequence from the structure"
        struc = s3
    lengths = [int(x) for x in a.lengths.split(",")]
    cases = {L: (full[:L], struc[:L] if struc else "#" * L) for L in lengths if L <= len(full)}

    def key(L):
        tag = "3di" if struc else "seq"
        return Path(a.ref_cache or ".", f"{a.model}_{L}_{tag}.npy")

    t0 = time.time()
    refs = {L: np.load(key(L)) for L in cases if a.ref_cache and key(L).is_file()}
    todo = [L for L in cases if L not in refs]
    if todo:
        ref_fn = (saprot_reference if is_saprot else esmc_reference)(a.model)
        for L in todo:
            refs[L] = ref_fn(*cases[L])
            if a.ref_cache:
                key(L).parent.mkdir(parents=True, exist_ok=True)
                np.save(key(L), refs[L])
        del ref_fn
        import gc
        gc.collect()
    t_ref = time.time() - t0
    if a.ref_only:
        print(f"references for {a.model} at {sorted(refs)} in {t_ref:.0f} s", flush=True)
        return

    if is_saprot:
        from tt_bio import saprot as mod
        model = mod.load_saprot(a.model, fast=a.fast)
        run = lambda aa, s: mod.embed_sequences(model, {"q": (aa, s)})[0].per_residue
    else:
        from tt_bio import esmc as mod
        model = mod.load_esmc(a.model, fast=a.fast)
        run = lambda aa, _s: mod.embed_sequences(model, {"q": aa})[0].per_residue

    with open(a.out, "a") as fh:
        for L, (aa, s) in cases.items():
            with during() as clk:
                t = time.time()
                dev = run(aa, s)
                wall = time.time() - t
            assert dev.shape[0] == L, f"device returned {dev.shape[0]} rows for {L} residues"
            row = {"model": a.model, "fast": a.fast, "L": L, "structure_tokens": bool(struc),
                   "n_3di_resolved": sum(c != "#" for c in s), "device_wall_s": round(wall, 1),
                   "aiclk": clk.summary(), "ref_wall_s_all": round(t_ref, 1),
                   "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                   **metrics(dev, refs[L])}
            print(json.dumps(row), flush=True)
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

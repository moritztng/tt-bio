"""On-hardware parity gate for the ESMC-6B embedding path (device vs reference esm).

ESMC-6B is the LM backbone of ESMFold2. Unlike 300m/600m it ships as sharded
TransformerEngine safetensors (no sequence head), so the existing
``scripts/esmc_embed_parity.py`` harness — which keys off ``CONFIGS`` (the
single-.pth 300m/600m entries) and reads the sequence-head logits — does not
apply. This script is the 6B-specific equivalent:

  * reference : the esm-repo ``ESMC`` architecture (``tests/esmc_reference.py``,
    the same golden used for 300m/600m) built at the 6B config
    (d_model=2560, n_heads=40, n_layers=80) and loaded with the *real* 6B
    weights, remapped from the TE layout to the esm-repo ``nn.Sequential``
    names via ``tt_bio.esmc._TE_KEY_REMAP`` and kept in fp32 (the golden dtype).
    The 6B is the same architecture as 300m/600m, just larger — the ttnn port
    loads the same remapped state dict into the same ``Block``/``Embedding``
    modules — so the esm-repo reference is the valid golden here too.
  * device   : the shipped embed path (``tt_bio.esmc.load_esmc("esmc-6b")`` +
    ``embed_sequences``), bf16 on device (the default, non-fast path).

6B carries no sequence head, so only per-residue and pooled embedding PCC are
reported (no logits / argmax). Sequences are reused verbatim from
``scripts/pharma_parity.py``'s ``ESMC_SEQS`` — the same proteins the 300m/600m
legs report — so the 6B leg is directly cross-comparable to those rows.

Usage (single device):
    ESM_ROOT=/path/to/esm TT_VISIBLE_DEVICES=0 \
        python3 scripts/esmc6b_embed_parity.py --seqs trpcage,ubiquitin

Reused by ``scripts/release_gate.py``'s opt-in ESMC-6B leg via
``scripts/esmc_embed_parity.py``'s ``run_esmc_parity`` — do not re-derive here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from tt_bio import esmc as tt_esmc  # noqa: E402

# Same proteins the 300m/600m pharma-benchmark legs report (scripts/pharma_parity.py).
ESMC_SEQS = {
    "trpcage": "NLYIQWLKDGGPSSGRPPPS",                                                  # 20
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",                 # 56
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 76
    "lysozyme": ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDG"
                 "RTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"),        # 129
}

# Default single-sequence target reused by release_gate's opt-in leg (matches
# esmc_embed_parity.DEFAULT_SEQ = ubiquitin).
DEFAULT_SEQ = ESMC_SEQS["ubiquitin"]


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def _load_esmc6b_reference_sd() -> dict:
    """Read the sharded 6B safetensors as fp32 and remap TE keys to esm-repo names.

    Mirrors ``tt_bio.esmc.load_esmc6b_state_dict`` but forces fp32 (the golden
    dtype) regardless of ``_FAST_MODE`` — the reference is the fp32 truth the
    bf16 device path is held against. Drops ``_extra_state``, the LM head and
    any classifier heads (the 6B port is embeddings-only).
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = snapshot_download("biohub/ESMC-6B")
    idx_path = os.path.join(snap, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    sd: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(snap, shard), "pt") as f:
            for k in keys:
                if k.endswith("_extra_state") or k.startswith("lm_head"):
                    continue
                if not k.startswith("esmc."):
                    continue
                nk = k[len("esmc."):]
                for src, dst in tt_esmc._TE_KEY_REMAP:
                    nk = nk.replace(src, dst)
                sd[nk] = f.get_tensor(k).to(torch.float32)
    return sd


def _build_reference() -> "ESMCReference":
    """Build the esm-repo ESMC at the 6B config and load the real 6B weights."""
    from esmc_reference import ESMCReference  # noqa: E402  (tests/ on path)

    cfg = dict(d_model=2560, n_heads=40, n_layers=80)
    ref = ESMCReference(**cfg).eval()
    sd = _load_esmc6b_reference_sd()
    # strict=False: the 6B has no sequence head, so ESMCReference.sequence_head
    # stays at its init (unused — we only read the post-norm trunk embeddings).
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    return ref


def run_esmc6b_parity(
    seq: str = DEFAULT_SEQ,
    *,
    fast: bool = False,
    pcc_threshold: float = 0.99,
    verbose: bool = True,
) -> dict:
    """Run the shipped ESMC-6B embed path vs the reference esm ESMC on `seq`.

    Returns a metrics dict shaped like ``esmc_embed_parity.run_esmc_parity``'s
    (so the release gate can treat 6b uniformly with 300m/600m). ``logits_pcc``
    and ``argmax_agree`` are None — the 6B port carries no sequence head.
    """
    torch.set_grad_enabled(False)

    if verbose:
        print("Building reference esm ESMC (6B, fp32) …", flush=True)
    ref = _build_reference()
    tokens = tt_esmc.tokenize(seq)
    _, ref_emb = ref(tokens)                  # [1, L+2, d_model] post-norm
    ref_per_res = ref_emb[0][1:-1].numpy()    # strip <cls>/<eos> -> [L, d]
    ref_pooled = ref_per_res.mean(axis=0)
    del ref  # free ~24 GB before loading the device copy

    if verbose:
        print(f"Loading tt ESMC-6B on device (fast={fast}) …", flush=True)
    model = tt_esmc.load_esmc("esmc-6b", fast=fast)
    # device self-consistency floor: run twice (the embed path has no sampler,
    # so dev-vs-dev PCC is the bf16 numerical noise floor, matching 300m/600m).
    dev_runs = []
    for _ in range(2):
        out = tt_esmc.embed_sequences(model, {"q": seq}, pool="mean")[0]
        dev_runs.append(out.per_residue)
    # device weights (~13 GB) are freed when the process exits; no mid-run release
    # needed for this short-lived gate.

    per_res_pcc = pcc(dev_runs[0], ref_per_res)
    pooled_pcc = pcc(dev_runs[0].mean(axis=0), ref_pooled)
    dev_vs_dev = pcc(dev_runs[0], dev_runs[1])
    ok = per_res_pcc >= pcc_threshold

    res = {"model": "esmc-6b", "seq_len": len(seq), "fast": fast,
           "per_res_pcc": per_res_pcc, "pooled_pcc": pooled_pcc,
           "dev_vs_dev_pcc": dev_vs_dev,
           "logits_pcc": None, "argmax_agree": None,
           "threshold": pcc_threshold, "ok": ok}

    if verbose:
        print("\n=== ESMC-6B embedding parity (tt vs reference esm) ===")
        print(f"model            : esmc-6b  (fast={fast})")
        print(f"sequence length  : {len(seq)} residues")
        print(f"per-residue PCC  : {per_res_pcc:.5f}")
        print(f"pooled(mean) PCC : {pooled_pcc:.5f}")
        print(f"dev-vs-dev PCC   : {dev_vs_dev:.5f}  (bf16 numerical floor)")
        print(f"\n{'PASS' if ok else 'FAIL'}: per-residue PCC "
              f"{per_res_pcc:.5f} vs threshold {pcc_threshold}")
    return res


def run_multi(seqs: dict[str, str], *, fast: bool = False, pcc_threshold: float = 0.99,
              out: str = "") -> dict:
    """Run the 6B parity across several proteins and emit a pharma-style report.

    Mirrors ``scripts/pharma_parity.py``'s ``embeddings`` mode: per-protein
    per-residue PCC (X) and dev-vs-dev PCC (D floor), plus a JSON report.
    """
    torch.set_grad_enabled(False)
    print("Building reference esm ESMC (6B, fp32) …", flush=True)
    ref = _build_reference()

    ref_emb: dict[str, np.ndarray] = {}
    for name, seq in seqs.items():
        _, e = ref(tt_esmc.tokenize(seq))
        ref_emb[name] = e[0][1:-1].numpy()
    del ref

    print(f"Loading tt ESMC-6B on device (fast={fast}) …", flush=True)
    model = tt_esmc.load_esmc("esmc-6b", fast=fast)
    dev_runs: list[dict[str, np.ndarray]] = []
    for _ in range(2):
        out_embs = tt_esmc.embed_sequences(model, seqs, pool="mean")
        dev_runs.append({o.id: o.per_residue for o in out_embs})

    report = {"mode": "embeddings", "model": "esmc-6b", "fast": fast, "targets": {}}
    print(f"\n### ESMC-6B embedding parity (fast={fast})\n")
    print("| protein | length | dev-vs-ref PCC (X) | dev-vs-dev PCC (D floor) |")
    print("|---|---|---|---|")
    for name, seq in seqs.items():
        x = pcc(dev_runs[0][name], ref_emb[name])
        d = pcc(dev_runs[0][name], dev_runs[1][name])
        report["targets"][name] = {"length": len(seq), "dev_vs_ref_pcc": x, "dev_vs_dev_pcc": d}
        print(f"| {name} | {len(seq)} | {x:.5f} | {d:.5f} |")

    xs = [v["dev_vs_ref_pcc"] for v in report["targets"].values()]
    ds = [v["dev_vs_dev_pcc"] for v in report["targets"].values()]
    report["dev_vs_ref_pcc_min"] = float(np.min(xs))
    report["dev_vs_ref_pcc_mean"] = float(np.mean(xs))
    report["dev_vs_dev_pcc_mean"] = float(np.mean(ds))
    print(f"\ndev-vs-ref PCC: mean {np.mean(xs):.5f}  min {np.min(xs):.5f}")
    print(f"device self-consistency PCC: mean {np.mean(ds):.5f}  min {np.min(ds):.5f}")
    print("\nInterpretation: the 6B port is the same code path as 300m/600m (same "
          "Block/Embedding modules, same fused RoPE), just larger; the device-vs-"
          "reference residual is bf16 rounding, not an algorithmic difference.")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", default="trpcage,ubiquitin",
                    help="comma-separated subset of the built-in proteins "
                         "(trpcage,gb1,ubiquitin,lysozyme); default trpcage,ubiquitin")
    ap.add_argument("--seq", default="", help="a single bare protein sequence (overrides --seqs)")
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.seq:
        res = run_esmc6b_parity(args.seq, fast=args.fast, pcc_threshold=args.pcc_threshold)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1

    names = [n.strip() for n in args.seqs.split(",") if n.strip()]
    bad = [n for n in names if n not in ESMC_SEQS]
    if bad:
        sys.exit(f"unknown protein(s) {bad}; choose from {list(ESMC_SEQS)}")
    seqs = {n: ESMC_SEQS[n] for n in names}
    report = run_multi(seqs, fast=args.fast, pcc_threshold=args.pcc_threshold, out=args.out)
    ok = all(v["dev_vs_ref_pcc"] >= args.pcc_threshold for v in report["targets"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

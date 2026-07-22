"""On-hardware parity gate for the standalone ESMC embedding API.

Runs the *shipped* embedding path (``tt_bio.esmc.load_esmc`` + ``embed_sequences``)
against the reference esm ESMC on real trained weights, and reports per-residue
embedding PCC — the accuracy metric that gates this capability. Also reports
pooled-embedding PCC and (300m/600m) sequence-head argmax agreement.

Usage (single device):
    ESM_ROOT=/path/to/esm TT_VISIBLE_DEVICES=0 \
        python3 scripts/esmc_embed_parity.py --model esmc-300m

The reference builder lives in tests/esmc_reference.py and needs the esm clone
on ESM_ROOT. Real weights download from the biohub HF repo on first use.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from tt_bio import esmc as tt_esmc  # noqa: E402

# Reuse the same protein set / sequences as the 6B multi-leg (scripts/esmc6b_embed_parity).
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from esmc6b_embed_parity import ESMC_SEQS  # noqa: E402  (scripts/ sibling)

# Human ubiquitin (76 aa) — a real, well-folded protein the LM is confident on.
DEFAULT_SEQ = ESMC_SEQS["ubiquitin"]


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def load_reference(name: str, sd: dict):
    """Build the reference esm ESMC for `name` and load `sd` into it."""
    from esmc_reference import ESMCReference  # noqa: E402  (tests/ on path)

    cfg = tt_esmc.CONFIGS[name][0]
    ref = ESMCReference(**cfg).eval()
    ref.load_state_dict(sd, strict=False)
    return ref


def run_esmc_parity(
    name: str,
    seq: str = DEFAULT_SEQ,
    *,
    fast: bool = False,
    pcc_threshold: float = 0.99,
    verbose: bool = True,
) -> dict:
    """Run the shipped ESMC embedding path vs the reference esm ESMC on `seq`.

    This is the on-hardware accuracy check that gates the ESMC embedding
    capability — including the fused-RoPE numerics change in ``esmc._rope``,
    which the bucketed embed path always takes (``BUCKET=64`` pads L to a
    tile-aligned length, so ``_rope`` selects ``ttnn.experimental.rotary_embedding``).

    Returns a metrics dict; ``ok`` is True iff per-residue PCC >= ``pcc_threshold``.
    Reused by ``scripts/release_gate.py``'s ESMC leg — do not re-derive here.

    ESMC-6B (the ESMFold2 LM backbone) ships as sharded TransformerEngine
    safetensors with no sequence head, so it cannot use the single-.pth / logits
    path below; it delegates to ``scripts/esmc6b_embed_parity.run_esmc6b_parity``
    (same esm-repo golden reference, fp32, just the 6B config + remapped weights).
    """
    if name == "esmc-6b":
        import importlib  # noqa: E402

        _scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from esmc6b_embed_parity import run_esmc6b_parity  # noqa: E402  (scripts/ sibling)

        return run_esmc6b_parity(seq, fast=fast, pcc_threshold=pcc_threshold, verbose=verbose)

    torch.set_grad_enabled(False)

    # --- real trained weights (downloads on first use) ---
    from huggingface_hub import hf_hub_download

    _cfg, repo_id, wpath = tt_esmc.CONFIGS[name]
    if verbose:
        print(f"Fetching {name} weights from {repo_id} …", flush=True)
    path = hf_hub_download(repo_id, wpath)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd

    # --- reference (CPU torch) ---
    if verbose:
        print("Building reference esm ESMC …", flush=True)
    ref = load_reference(name, sd)
    tokens = tt_esmc.tokenize(seq)
    ref_logits, ref_emb = ref(tokens)             # [1,L+2,64], [1,L+2,d]
    ref_per_res = ref_emb[0][1:-1].numpy()        # strip <cls>/<eos>
    ref_pooled = ref_per_res.mean(axis=0)
    ref_logits_res = ref_logits[0][1:-1].numpy()

    # --- shipped tt embedding API (load_esmc downloads + loads the same .pth) ---
    if verbose:
        print(f"Loading tt ESMC on device{' (fast)' if fast else ''} …", flush=True)
    model = tt_esmc.load_esmc(name, fast=fast)
    emb = tt_esmc.embed_sequences(model, {"ubq": seq},
                                  return_logits=True, pool="mean")[0]

    per_res_pcc = pcc(emb.per_residue, ref_per_res)
    pooled_pcc = pcc(emb.pooled, ref_pooled)
    logits_pcc = pcc(emb.logits, ref_logits_res)
    argmax_agree = float((emb.logits.argmax(-1) == ref_logits_res.argmax(-1)).mean())
    ok = per_res_pcc >= pcc_threshold

    res = {"model": name, "seq_len": len(seq), "fast": fast,
           "per_res_pcc": per_res_pcc, "pooled_pcc": pooled_pcc,
           "logits_pcc": logits_pcc, "argmax_agree": argmax_agree,
           "threshold": pcc_threshold, "ok": ok}

    if verbose:
        print("\n=== ESMC embedding parity (tt vs reference esm) ===")
        print(f"model            : {name}  (fast={fast})")
        print(f"sequence length  : {len(seq)} residues")
        print(f"per-residue PCC  : {per_res_pcc:.5f}")
        print(f"pooled(mean) PCC : {pooled_pcc:.5f}")
        print(f"logits PCC       : {logits_pcc:.5f}")
        print(f"argmax agreement : {argmax_agree:.4f}")
        print(f"\n{'PASS' if ok else 'FAIL'}: per-residue PCC "
              f"{per_res_pcc:.5f} vs threshold {pcc_threshold}")
    return res


def run_multi(
    name: str,
    seqs: dict[str, str],
    *,
    fast: bool = False,
    pcc_threshold: float = 0.99,
    out: str = "",
) -> dict:
    """Run the shipped ESMC embedding path vs the reference across several proteins.

    Mirrors ``scripts/esmc6b_embed_parity.run_multi``: one reference build + one device
    load, two device passes, then per-protein per-residue PCC (X) and dev-vs-dev PCC
    (D floor). Emits a pharma-style report whose shape matches the committed
    ``docs/implementation-parity-data/esmc-{300m,600m}.json`` (``targets`` dict), so
    ``scripts/full_parity_gate.py``'s ``_esmc_verdict`` (min per-res PCC >= 0.99) reads
    it directly. Only 300m/600m here; 6b stays in ``esmc6b_embed_parity``.
    """
    import json  # noqa: E402  (local import keeps the single-seq path import-light)

    from huggingface_hub import hf_hub_download  # noqa: E402

    torch.set_grad_enabled(False)

    if verbose := True:
        print(f"Fetching {name} weights …", flush=True)
    _cfg, repo_id, wpath = tt_esmc.CONFIGS[name]
    path = hf_hub_download(repo_id, wpath)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd

    print("Building reference esm ESMC …", flush=True)
    ref = load_reference(name, sd)
    ref_per_res: dict[str, np.ndarray] = {}
    for pname, seq in seqs.items():
        _, e = ref(tt_esmc.tokenize(seq))
        ref_per_res[pname] = e[0][1:-1].numpy()
    del ref

    print(f"Loading tt ESMC on device (fast={fast}) …", flush=True)
    model = tt_esmc.load_esmc(name, fast=fast)
    dev_runs: list[dict[str, np.ndarray]] = []
    for _ in range(2):
        out_embs = tt_esmc.embed_sequences(model, seqs, pool="mean")
        dev_runs.append({o.id: o.per_residue for o in out_embs})

    report = {"mode": "embeddings", "model": name, "fast": fast, "targets": {}}
    print(f"\n### ESMC {name} embedding parity (fast={fast})\n")
    print("| protein | length | dev-vs-ref PCC (X) | dev-vs-dev PCC (D floor) |")
    print("|---|---|---|---|")
    for pname, seq in seqs.items():
        x = pcc(dev_runs[0][pname], ref_per_res[pname])
        d = pcc(dev_runs[0][pname], dev_runs[1][pname])
        report["targets"][pname] = {"length": len(seq), "dev_vs_ref_pcc": x, "dev_vs_dev_pcc": d}
        print(f"| {pname} | {len(seq)} | {x:.5f} | {d:.5f} |")

    xs = [v["dev_vs_ref_pcc"] for v in report["targets"].values()]
    ds = [v["dev_vs_dev_pcc"] for v in report["targets"].values()]
    report["dev_vs_ref_pcc_min"] = float(np.min(xs))
    report["dev_vs_ref_pcc_mean"] = float(np.mean(xs))
    report["dev_vs_dev_pcc_mean"] = float(np.mean(ds))
    print(f"\ndev-vs-ref PCC: mean {np.mean(xs):.5f}  min {np.min(xs):.5f}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.MODELS))
    ap.add_argument("--seq", default="", help="a single bare protein sequence (single-seq mode)")
    ap.add_argument("--seqs", default="",
                    help="comma-separated subset of trpcage,gb1,ubiquitin,lysozyme "
                         "(multi-leg mode; produces the pharma-style targets report)")
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="", help="write the multi-leg JSON report here")
    args = ap.parse_args()

    if args.seqs:
        names = [n.strip() for n in args.seqs.split(",") if n.strip()]
        bad = [n for n in names if n not in ESMC_SEQS]
        if bad:
            sys.exit(f"unknown protein(s) {bad}; choose from {list(ESMC_SEQS)}")
        if args.model == "esmc-6b":
            # 6b has its own multi-leg harness (different reference build); delegate.
            from esmc6b_embed_parity import run_multi as run_6b_multi  # noqa: E402
            seqs = {n: ESMC_SEQS[n] for n in names}
            report = run_6b_multi(seqs, fast=args.fast, pcc_threshold=args.pcc_threshold, out=args.out)
            ok = all(v["dev_vs_ref_pcc"] >= args.pcc_threshold for v in report["targets"].values())
            return 0 if ok else 1
        seqs = {n: ESMC_SEQS[n] for n in names}
        report = run_multi(args.model, seqs, fast=args.fast,
                           pcc_threshold=args.pcc_threshold, out=args.out)
        ok = all(v["dev_vs_ref_pcc"] >= args.pcc_threshold for v in report["targets"].values())
        return 0 if ok else 1

    seq = args.seq or DEFAULT_SEQ
    res = run_esmc_parity(args.model, seq,
                         fast=args.fast, pcc_threshold=args.pcc_threshold)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

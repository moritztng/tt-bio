#!/usr/bin/env python3
"""ESM-C and SaProt embedding leg of the perf-page GPU benchmark.

The six embedding rows are not folds, so gpu5_bench.py's fold loop does not fit them. This is
the same timing scope in the shape an encoder forward has: weights loaded once, the forward
wrapped in torch.cuda.synchronize() on both sides, pass 1 discarded as cold, then N warm
passes reported as a median with n / min / max / spread. It emits the JSON shape gpu5_bench.py
emits so every row lands in one table.

Three things this harness pins that a naive script gets wrong, each a fairness question rather
than a convenience:

  * BATCH. `tt-bio embed --batch_size 8` is the shipped default, but the shipped batcher caps a
    batch at batch_size * 512 tokens, and a 512 aa sequence buckets to 576, so the batch that
    actually executes is 7. The GPU side has to run 7 as well or the two numbers are not the
    same job. --batch defaults per row to what the Tenstorrent side executes and the count is
    written into the result.
  * DTYPE. Both repos' config.json says float32; both sides of the page run bf16. bf16 is the
    faster arm for the GPU, not the slower one, and it is what ESMFold2's already-published GPU
    cells run this same ESMC-6B backbone in. Detuning the reference to fp32 would inflate our
    ratio. The fp32 arm still runs, once, as the precision control below.
  * ATTENTION. transformers picks eager attention when sdpa is not asked for, which would
    under-report the GPU. sdpa is requested explicitly, the resolved implementation is
    recorded, and the SDPA call counter has to be non-zero -- a requested backend is not proof
    a backend ran.

The precision control is a cosine similarity between the bf16 pooled vector and the fp32 pooled
vector of the same weights on the same input, on the same box. It is a self-consistency check
on the timed arm's precision, not a cross-implementation parity claim: the pooling here need not
match tt_bio's pooling for the check to mean what it says.

Usage:
    /root/venv-esm312/bin/python gpu_embed_bench.py --model esmc-300m \\
        --seq-file fixtures/prot512.seq --out /root/results/gpu_esmc-300m_prot512_b200.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Per-row: HF repo, loader class, and the batch the Tenstorrent side executes at 512 aa.
# esmc-6b runs one sequence at a time because the sharded 6B backbone ignores batch_size
# (docs/perf_baselines.json's note says so and the shipped perf gate gates it separately), so
# its matched batch is 1 and calling that "batch 8" would be a false match.
ROWS = {
    "esmc-300m":   dict(repo="biohub/ESMC-300M", loader="auto",   batch=7, family="esmc"),
    "esmc-600m":   dict(repo="biohub/ESMC-600M", loader="auto",   batch=7, family="esmc"),
    "esmc-6b":     dict(repo="biohub/ESMC-6B",   loader="auto",   batch=1, family="esmc"),
    "saprot-35m":  dict(repo="westlake-repl/SaProt_35M_AF2",  loader="esm", batch=7, family="saprot"),
    "saprot-650m": dict(repo="westlake-repl/SaProt_650M_AF2", loader="esm", batch=7, family="saprot"),
    "saprot-1.3b": dict(repo="westlake-repl/SaProt_1.3B_AF2", loader="esm", batch=7, family="saprot"),
}

PROT512_SHA = "141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d"
COSINE_BAR = 0.999


def disable_cudnn_sdpa() -> bool:
    """Drop torch SDPA's cuDNN backend when it has no plan for this GPU.

    On sm_100 (B200) with torch 2.11+cu130 / cuDNN 9.19, SDPA dispatches to the cuDNN backend
    and dies with "cudnn_frontend Error: No valid execution plans built" -- cuDNN ships no
    attention plan for Blackwell at these shapes. This is NOT a detune: the backend cannot
    execute at all, so switching it off is what lets SDPA reach a backend that runs (flash /
    mem-efficient / math). The row still runs torch SDPA, which is what the manifest specifies
    and what the torch_sdpa counter proves. Returns whether the knob was actually flipped, so
    the result JSON can record it as provenance rather than leaving it implicit.
    """
    torch = importlib.import_module("torch")
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            torch.backends.cuda.enable_cudnn_sdp(False)
            return True
    except Exception:
        pass
    return False


def install_sdpa_counter() -> dict:
    """Count torch SDPA calls. A model with 0 here ran eager attention, whatever was asked for."""
    counts = {"torch_sdpa": 0}
    torch = importlib.import_module("torch")
    orig = torch.nn.functional.scaled_dot_product_attention

    def wrapper(*a, **kw):
        counts["torch_sdpa"] += 1
        return orig(*a, **kw)
    torch.nn.functional.scaled_dot_product_attention = wrapper
    return counts


def summarize(times: list[float], label: str = "pass") -> dict:
    """Pass 1 is the cold/compile pass and is discarded explicitly."""
    if not times:
        return dict(error=f"no {label} was timed")
    cold, warm = times[0], times[1:]
    if not warm:
        return dict(cold_s=round(cold, 4), error=f"no warm {label}")
    ts = sorted(warm)
    return dict(
        cold_s=round(cold, 4),
        warm_times_s=[round(t, 4) for t in warm],
        warm_n=len(warm),
        warm_min_s=round(ts[0], 4),
        warm_median_s=round(ts[len(ts) // 2], 4),
        warm_max_s=round(ts[-1], 4),
        warm_spread_pct=round(100.0 * (ts[-1] - ts[0]) / ts[len(ts) // 2], 2),
    )


def build_inputs(family: str, seq: str, batch: int, tok):
    """One batch of `batch` identical copies of the fixture sequence.

    SaProt's vocabulary is fused AA+3Di: each residue is one token spelled as two characters,
    and sequence-only mode spells the structure half '#'. That is what tt_bio.saprot.tokenize
    builds (`_TOK_TO_IDX[a + s]`) and it is the shipped default when no --structure is given, so
    the GPU side spells it the same way. Identical copies rather than distinct sequences because
    the Tenstorrent batcher buckets by length: distinct 512 aa sequences bucket identically, so
    the padding, and therefore the work, is the same either way.
    """
    text = seq if family == "esmc" else "".join(a + "#" for a in seq)
    enc = tok([text] * batch, return_tensors="pt")
    return enc


def pooled(hidden):
    """Mean over residues, dropping the BOS and EOS positions -- tt_bio's `mean` pool over
    `per_residue = emb[1:-1]`."""
    return hidden[:, 1:-1, :].float().mean(dim=1)


def load_model(spec: dict, dtype, attn: str):
    """Load a row's weights at the requested dtype, and prove the dtype took.

    `from_pretrained` swallows an unrecognised kwarg rather than raising, and transformers
    renamed this argument (`torch_dtype` -> `dtype`). On a version that does not know the name
    we asked for, the model loads fp32 in silence: nothing crashes, the cosine control reads a
    perfect 1.0 because both arms are the same arm, and the row publishes an fp32 time under a
    bf16 label. So the loaded dtype is asserted, not requested -- the same reason the SDPA
    backend is counted instead of trusted.
    """
    transformers = importlib.import_module("transformers")
    cls = getattr(transformers, "AutoModelForMaskedLM" if spec["loader"] == "auto"
                  else "EsmForMaskedLM")
    try:
        model = cls.from_pretrained(spec["repo"], dtype=dtype, attn_implementation=attn)
    except TypeError:  # pre-rename transformers
        model = cls.from_pretrained(spec["repo"], torch_dtype=dtype, attn_implementation=attn)
    got = next(p.dtype for p in model.parameters() if p.is_floating_point())
    if got != dtype:
        raise SystemExit(f"{spec['repo']}: asked for {dtype}, loaded {got}. Refusing to "
                         f"publish a {got} timing as {dtype}.")
    return model


def load_tokenizer(spec: dict):
    transformers = importlib.import_module("transformers")
    # SaProt ships a vocab.txt for the fused 446-token vocabulary, which is EsmTokenizer's
    # format; AutoTokenizer resolves it from the repo's tokenizer_config.json either way.
    return transformers.AutoTokenizer.from_pretrained(spec["repo"])


def run(args) -> dict:
    torch = importlib.import_module("torch")
    spec = ROWS[args.model]

    seq = args.seq_file.read_text().strip()
    fixture_sha = hashlib.sha256(args.seq_file.read_bytes()).hexdigest()
    if args.expect_sha and fixture_sha != args.expect_sha:
        raise SystemExit(f"fixture sha256 {fixture_sha} != expected {args.expect_sha}")

    cudnn_sdp_disabled = disable_cudnn_sdpa()
    sdpa = install_sdpa_counter()
    tok = load_tokenizer(spec)

    load_t0 = time.perf_counter()
    model = load_model(spec, torch.bfloat16, args.attn).cuda().eval()
    load_s = time.perf_counter() - load_t0

    enc = build_inputs(spec["family"], seq, args.batch, tok)
    n_tokens = int(enc["input_ids"].shape[1])
    enc = {k: v.cuda() for k, v in enc.items()}

    times, last = [], None
    for _ in range(args.repeat + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        last = out.hidden_states[-1]
    bf16_pool = pooled(last)[0].cpu()
    d_model = int(last.shape[-1])
    peak_bf16 = torch.cuda.max_memory_allocated()

    # Precision control: same weights, same input, fp32, one pass. Not timed.
    del model, out, last
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ctl = load_model(spec, torch.float32, args.attn).cuda().eval()
    with torch.no_grad():
        fp32_pool = pooled(ctl(**enc, output_hidden_states=True).hidden_states[-1])[0].cpu()
    peak_fp32 = torch.cuda.max_memory_allocated()
    del ctl
    torch.cuda.empty_cache()

    cos = float(torch.nn.functional.cosine_similarity(
        bf16_pool.float().unsqueeze(0), fp32_pool.float().unsqueeze(0)).item())
    finite = bool(torch.isfinite(bf16_pool).all())

    s = summarize(times)
    med = s.get("warm_median_s")
    # Derive seq/s from the batch that actually ran, not the one we asked for. The whole point
    # of this row set is that requested and executed differ at 512 aa.
    n_seqs = int(enc["input_ids"].shape[0])
    return dict(
        s,
        load_s=round(load_s, 2),
        repo=spec["repo"], family=spec["family"], loader=spec["loader"],
        batch_requested=args.batch, batch_executed=int(enc["input_ids"].shape[0]),
        n_residues=len(seq), n_tokens=n_tokens, d_model=d_model,
        seq_per_s=(round(n_seqs / med, 3) if med else None),
        s_per_seq=(round(med / n_seqs, 5) if med else None),
        kernel_counts_total=dict(sdpa),
        attn_implementation_requested=args.attn,
        cudnn_sdpa_disabled=cudnn_sdp_disabled,
        # A requested backend is not a running backend. Zero SDPA calls means eager attention.
        attn_engaged=bool(sdpa["torch_sdpa"] > 0),
        accuracy=dict(cosine_bf16_vs_fp32=round(cos, 6), bar=COSINE_BAR,
                      pass_=bool(cos >= COSINE_BAR and finite), all_finite=finite,
                      pooled_norm=round(float(bf16_pool.float().norm()), 4)),
        peak_mem_bf16_mib=round(peak_bf16 / 2**20, 1),
        peak_mem_fp32_mib=round(peak_fp32 / 2**20, 1),
        fixture=dict(path=str(args.seq_file), sha256=fixture_sha),
    )


def dry_run(args) -> int:
    """The CPU half of the check: fixture digest, batch, and the token count the row will feed.

    Token count is arithmetic off the shipped tokenizers rather than a guess: both tt_bio.esmc
    and tt_bio.saprot build [BOS] + one token per residue + [EOS], and SaProt's fused AA+3Di
    token is two characters that tokenize to one id, so both families land on len(seq) + 2. If
    a real tokenizer on the box disagrees with this number, that is a finding, not a rounding.
    """
    spec = ROWS[args.model]
    seq = args.seq_file.read_text().strip()
    sha = hashlib.sha256(args.seq_file.read_bytes()).hexdigest()
    text = seq if spec["family"] == "esmc" else "".join(a + "#" for a in seq)
    ok = (sha == args.expect_sha) and len(seq) == 512
    print(json.dumps(dict(
        model=args.model, repo=spec["repo"], loader=spec["loader"], family=spec["family"],
        batch=args.batch, n_residues=len(seq), expected_n_tokens=len(seq) + 2,
        model_input_chars=len(text), fixture_sha256=sha,
        fixture_sha_matches=(sha == args.expect_sha),
        attn=args.attn, dtype="bfloat16", dry_run=True), indent=2))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(ROWS))
    ap.add_argument("--repeat", type=int, default=3, help="warm passes; pass 1 is cold on top")
    ap.add_argument("--seq-file", type=Path, default=HERE / "fixtures/prot512.seq")
    ap.add_argument("--expect-sha", default=PROT512_SHA,
                    help="fail rather than measure if the fixture is not byte-identical")
    ap.add_argument("--batch", type=int, default=None,
                    help="default: the batch the Tenstorrent side executes for this row")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--dry-run", action="store_true",
                    help="no torch, no weights: verify the fixture digest, the batch this row "
                         "runs and the token count it will feed the model. Runs on any CPU box, "
                         "so a wrong fixture or a wrong batch is never found on a paid instance")
    ap.add_argument("--out", type=Path,
                    help="where the result JSON goes; not needed with --dry-run, which writes "
                         "nothing")
    args = ap.parse_args()
    if args.batch is None:
        args.batch = ROWS[args.model]["batch"]

    if args.dry_run:
        return dry_run(args)
    if args.out is None:
        ap.error("--out is required unless --dry-run")

    t0 = time.perf_counter()
    try:
        res, err = run(args), None
    except Exception:
        import traceback
        res, err = {}, traceback.format_exc()
        print(err, file=sys.stderr)

    torch = importlib.import_module("torch")
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    pkgs = {}
    for p in ("torch", "transformers", "esm", "tokenizers"):
        try:
            pkgs[p] = importlib.metadata.version(p)
        except Exception:
            pass

    summary = dict(
        model=args.model, side="gpu", unit="s per batch (batch stated per row)",
        gpu=torch.cuda.get_device_name(0),
        gpu_capability=list(torch.cuda.get_device_capability()),
        host_cpu=cpu, cpu_count=os.cpu_count(),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        dtype="bfloat16", packages=pkgs,
        session_wall_s=round(time.perf_counter() - t0, 1),
        error=err, result=res, date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    ok = err is None and res.get("accuracy", {}).get("pass_") and res.get("attn_engaged")
    return 0 if ok else 1


if __name__ == "__main__":
    import importlib.metadata  # noqa: F401  (populated lazily above)
    sys.exit(main())

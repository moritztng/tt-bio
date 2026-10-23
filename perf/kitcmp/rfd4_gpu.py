"""RFD4-Proteina on an NVIDIA GPU, timed under state/rfd4/PROTOCOL.md.

One process per arm: the model is loaded once and `rfd4.generate` is called once per batch, so batch 0 pays the
`torch.compile` warm-up and CUDA-graph capture (recorded as `"kind":"cold"`) and the rest are warm. Every batch
writes one evidence line to `--out/runs.jsonl` as it finishes, so a box that dies mid-arm still leaves its
measured batches behind.

    python rfd4_gpu.py --spec pdl1-256.yaml --size 256 --arm compile-cueq --batch 4 --warm 5 --out DIR \
                       [--seed 1234] [--set nn.attention_type=cuequivariance ...]

The recipe (`recycle_strategy=thinking_medium`, 60 steps, 180 trunk passes, bf16-mixed, `'~metrics'`) is fixed
here and is NOT settable from the command line: an arm that does less work is not an arm. `--set` carries only
options that change how the same work runs (backends, compile mode, dataloader workers).

`nn.*` is a BUILD-time setting, not a generate-time one: upstream raises
`ValueError: nn.* overrides apply when the model is built; a supplied model cannot take them`
(`rfproteina/generate.py:417`) for any `nn.*` reaching `generate()` with a model already loaded. So `--set
nn.k=v` is routed to `Proteina.from_pretrained(nn_overrides=...)` and everything else to `generate()`. That
cost a whole rented box on 2026-10-21: three of seven arms (compile-reduce-overhead, compile-cueq,
attn-torch) are defined purely by `nn.*` and could never run.
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

RECIPE = ["recycle_strategy=thinking_medium", "precision=bf16-mixed", "~metrics", "skip_existing=false"]
# `configs/inference.yaml` pins the Hub commit it was released against; from_pretrained does not read that
# config, so the pin is repeated here. An unpinned baseline is not a reproducible one.
HF_CKPT = os.environ.get("RFD4_CKPT", "nvidia/RFD4-Proteina")
HF_REV = os.environ.get("RFD4_CKPT_REV", "a06e2509c7033b010fc43614e9fd840303588502")
# rfproteina/nn/local_latents_transformer.py:45 — anything else is rejected at build time.
NN_RUNTIME_OPTIONS = ("torch_compile_mode", "torch_compile_fullgraph", "tri_mult_backend", "attention_type")


def split_overrides(sets):
    """(nn_overrides for from_pretrained, generate overrides). Imported by the box's pre-flight gate."""
    nn, gen = {}, []
    for s in sets:
        if not s.startswith("nn."):
            gen.append(s)
            continue
        key, _, val = s[3:].partition("=")
        if key not in NN_RUNTIME_OPTIONS:
            raise SystemExit(f"nn.{key} cannot change at inference; upstream allows {NN_RUNTIME_OPTIONS}")
        nn[key] = {"true": True, "false": False}.get(val.lower(), val)
    return nn, gen



def smi(fields):
    q = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    vals = q.stdout.strip().splitlines()[0].split(", ") if q.returncode == 0 and q.stdout.strip() else []
    out = {}
    for k, v in zip(fields, vals):
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v.strip()
    return out


def emit(rec):
    rec["t"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with runs.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def main():
    global runs
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path, help="condition spec from state/rfd4/inputs/")
    # Any shipped compile bucket: the gated 128/256/512 and the realistic-binder sizes rfd4-size-typical adds.
    ap.add_argument("--size", required=True, type=int, choices=tuple(range(128, 1025, 16)))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--batch", required=True, type=int)
    ap.add_argument("--warm", type=int, default=5, help="warm batches after the cold one")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--set", action="append", default=[], metavar="K=V", help="a Hydra override for this arm")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = a.out / "runs.jsonl"
    gpu = smi(["name", "clocks.sm", "power.draw", "utilization.gpu", "memory.total"])
    base = {"size": a.size, "arm": a.arm, "batch": a.batch, "seed": a.seed,
            "gpu": gpu.get("name", "unknown"), "sha": os.environ.get("RFD4_SHA", ""),
            "raw": str(a.out), "host": os.environ.get("RFD4_HOST", ""), "overrides": a.set}

    nn_overrides, gen_sets = split_overrides(a.set)
    base["nn_overrides"] = nn_overrides

    import rfd4_proteina as rfd4  # noqa: E402  (after argparse: importing torch is slow)

    t0 = time.perf_counter()
    rev = None if Path(HF_CKPT).exists() else HF_REV  # a local bundle has no Hub revision
    model = rfd4.Proteina.from_pretrained(HF_CKPT, revision=rev, nn_overrides=nn_overrides or None)
    emit({**base, "kind": "cold", "phase": "load", "designs": 0, "wall_s": round(time.perf_counter() - t0, 3)})

    for i in range(a.warm + 1):
        ov = RECIPE + [f"inputs={a.spec}", f"out_dir={a.out}/batch{i}", f"num_replicates={a.batch}",
                       f"batch_size={a.batch}", f"seed={a.seed + i}"] + gen_sets
        t0 = time.perf_counter()
        err = None
        try:
            rfd4.generate(model, overrides=ov)
        except Exception as e:  # an arm that fails to run is reported FAILED, never dropped
            err = f"{type(e).__name__}: {e}"
        wall = round(time.perf_counter() - t0, 3)
        made = len(list((a.out / f"batch{i}").rglob("*_generated.cif*"))) if (a.out / f"batch{i}").exists() else 0
        live = smi(["clocks.sm", "power.draw", "utilization.gpu"])
        rec = {**base, "kind": "cold" if i == 0 else "warm", "designs": made, "wall_s": wall,
               "sm_mhz": live.get("clocks.sm"), "power_w": live.get("power.draw"), "util_pct": live.get("utilization.gpu")}
        if i == 0:
            rec["phase"] = "compile"
        if err or made != a.batch:
            rec["status"] = "FAILED"
            rec["error"] = err or f"wrote {made} designs, expected {a.batch}"
        emit(rec)
        if err:
            sys.exit(1)


if __name__ == "__main__":
    main()

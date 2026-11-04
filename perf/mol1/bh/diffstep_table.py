"""Per-family table of one diffusion step from diffstep.py's ops.jsonl: ms, FLOP, bytes, TFLOP/s, GB/s, roof.

    python3 diffstep_table.py <out-dir> [--steps 200] [--json]

Families come from each op's tt_bio call chain. Bytes are every device operand read plus every output written, at its
padded tile size, so they are what an unfused op moves through DRAM at least once. Roof per op is max(FLOP / peak
math, bytes / peak DRAM) with the p150a figures measured in the ttnn-perf-profiling skill (HiFi4 bf16 100.5 TFLOP/s;
fp32 operands at HiFi4 take two passes, 50 TFLOP/s; 435 GB/s through ttnn). Gap s/fold = (ms - roof ms) x steps.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PEAK_BF16, PEAK_F32, PEAK_BW = 100.5e12, 50.25e12, 435e9


def family(o):
    ch = " ".join(o["chain"])
    fam = o["fam"]
    if "dit_shoist" in fam:
        return "DiT adaLN/gate conditioning (shoist)"
    if "dit" in fam:
        if "AttentionPairBias" in ch:
            return "DiT attention"
        if "AdaLN" in ch:
            return "DiT adaLN"
        if "gated" in ch:
            return "DiT gates"
        return "DiT transition + glue"
    for f, name in (("atom_enc", "atom encoder"), ("atom_dec", "atom decoder")):
        if f in fam:
            if "_attention" in ch or "_windows" in ch or "_merge_heads" in ch or "_lin_g" in ch:
                return f"{name} attention"
            if "_adaln" in ch or "AdaLN" in ch:
                return f"{name} adaLN"
            return f"{name} transition + glue"
    if "_cond_transitions" in ch or "Transition" in ch:
        return "conditioning transitions"
    return "denoise glue (cond, pool, in/out linears)"


def flop(o):
    op, ins = o["op"], [i for i in o["ins"] if i]
    if op in ("linear", "matmul", "experimental.minimal_matmul") and len(ins) >= 2:
        a, b = ins[0]["shape"], ins[1]["shape"]
        return 2 * math.prod(a[:-1]) * a[-1] * b[-1]
    if "scaled_dot_product_attention" in op and len(ins) >= 3:
        q, k = ins[0]["shape"], ins[1]["shape"]
        return 4 * math.prod(q[:-1]) * k[-2] * q[-1]
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = Path(a.out)
    ops = [json.loads(x) for x in (d / "ops.jsonl").read_text().splitlines()]
    whole = json.loads((d / "whole.json").read_text())
    rows = defaultdict(lambda: defaultdict(float))
    for o in ops:
        r = rows[family(o)]
        ms = o["ms"] or 0.0
        fl = flop(o)
        by = sum(i["bytes"] for i in o["ins"] + o["outs"] if i and i.get("dev"))
        f32 = any(i and i["dtype"] == "FLOAT32" for i in o["ins"][:2])
        roof = max(fl / (PEAK_F32 if f32 else PEAK_BF16), by / PEAK_BW) * 1e3 if o["ms"] is not None else 0.0
        r["n"] += 1
        r["ms"] += ms
        r["flop"] += fl
        r["bytes"] += by
        r["roof_ms"] += roof
        r["mm_ms"] += ms if fl else 0.0
    tot = sum(r["ms"] for r in rows.values())
    lines = ["| family | ops | ms/step | share | GFLOP | MB | TFLOP/s | GB/s | roof ms | gap s/fold |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(rows.items(), key=lambda kv: -kv[1]["ms"]):
        tf = r["flop"] / (r["mm_ms"] * 1e-3) / 1e12 if r["mm_ms"] else 0
        lines.append(f"| {k} | {int(r['n'])} | {r['ms']:.2f} | {100 * r['ms'] / tot:.1f} % | {r['flop'] / 1e9:.1f} | "
                     f"{r['bytes'] / 1e6:.0f} | {tf:.1f} | {r['bytes'] / (r['ms'] * 1e-3) / 1e9 if r['ms'] else 0:.0f} | "
                     f"{r['roof_ms']:.2f} | {(r['ms'] - r['roof_ms']) * a.steps / 1e3:.2f} |")
    lines.append(f"| sum of ops | {len(ops)} | {tot:.2f} | | | | | | | |")
    for f, v in whole.items():
        lines.append(f"| whole `{f}` replay ({len(v)} calls) | | {sum(x['ms'] or 0 for x in v):.2f} | | | | | | | |")
    print("\n".join(lines))
    top = sorted((o for o in ops if o["ms"]), key=lambda o: -o["ms"])[:25]
    print("\n| op | family | ms | in shapes | dtype |\n|---|---|---|---|---|")
    for o in top:
        print(f"| {o['op']} | {family(o)} | {o['ms']:.3f} | {' '.join(str(i['shape']) for i in o['ins'] if i)} | "
              f"{','.join(i['dtype'] for i in o['ins'] if i)} |")
    skipped = [o for o in ops if o["ms"] is None]
    print(f"\nnot re-run: {len(skipped)} ops ({', '.join(sorted({o['op'] + ':' + (o['err'] or '')[:40] for o in skipped})[:12])})")


if __name__ == "__main__":
    main()

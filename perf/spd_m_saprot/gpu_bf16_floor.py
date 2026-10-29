"""How far the GPU reference's own precision sits from fp32, on the proteins the SaProt levers are graded on.

The A100 cell runs SaProt in bf16 with torch SDPA (scripts/gpu_vs_tt/gpu_embed_bench.py). This runs the same HF
checkpoint in bf16 + sdpa on the CPU (fp32 accumulation, bf16 storage, as cuBLAS / flash do) and scores it against
the fp32 reference with the metric every TT lever is graded by (perf/mgx_embed/accuracy.metrics).

    python3 perf/spd_m_saprot/gpu_bf16_floor.py --model saprot-650m --pdb perf/bgx_size/targets/hPCSK9.pdb \
        --ref-cache ../refcache --out floor.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mgx_embed"))
import accuracy as acc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", action="append", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--ref-cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from transformers import EsmForMaskedLM
    from tt_bio import saprot, weights
    path = weights.fetch(a.model)
    f32 = bf = None
    for pdb in a.pdb:
        aa = acc.pdb_sequence(pdb)[: a.length]
        key = Path(a.ref_cache, f"{a.model}_{a.length}_{Path(pdb).stem}_seq.npy")
        ids = saprot.tokenize(aa, "#" * len(aa))
        if key.is_file():
            ref = np.load(key)
        else:
            f32 = f32 or EsmForMaskedLM.from_pretrained(path, dtype=torch.float32).eval()
            ref = f32.esm(ids).last_hidden_state[0][1:-1].float().numpy()
            key.parent.mkdir(parents=True, exist_ok=True)
            np.save(key, ref)
        bf = bf or EsmForMaskedLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        assert next(bf.parameters()).dtype == torch.bfloat16
        out = bf.esm(ids).last_hidden_state[0][1:-1].float().numpy()
        row = {"model": a.model, "pdb": Path(pdb).stem, "L": len(aa), "arm": "hf_bf16_sdpa_cpu",
               "attn": bf.config._attn_implementation, "vs_fp32": acc.metrics(out, ref)}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does SaProt's attention run slower on the q, k, v the block hands it than on copies of the same values?

ls138 (p300c at 1350 MHz, op_census, 650M page batch): shsdpa reads 11.6 ms over 33 calls in --fast at HiFi2 and 7.5 ms
in normal mode at the same config and shape. --fast's heads come from mm_rope, normal's from nlp_create_qkv_heads and
rotary_embedding. This catches the first attention call of a warm forward and times it, synchronised, on the block's
own tensors and on DRAM copies made by ttnn.clone and by a host round trip, at the config the block passed and at HiFi2.

    python perf/spd_m_saprot/sdpa_src.py --model saprot-650m --fast --out src.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-seqs", type=int, default=7)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import saprot as S
    model = S.load_saprot(a.model, fast=a.fast)
    dev = model.tt_device
    seq = Path(a.seq_file).read_text().strip()
    seqs = {f"s{i}": (seq, "#" * len(seq)) for i in range(a.n_seqs)}
    for _ in range(2):
        S.embed_sequences(model, seqs, batch_size=8)
    ttnn.synchronize_device(dev)

    orig, res = S.lm_sdpa, {}

    def timed(q, k, v, mask, kw):
        ttnn.synchronize_device(dev)
        ms = []
        for _ in range(a.calls):
            t = time.perf_counter()
            o = orig(q, k, v, mask, **kw)
            ttnn.synchronize_device(dev)
            ms.append(round((time.perf_counter() - t) * 1000, 4))
            ttnn.deallocate(o)
        return {"min": min(ms), "med": sorted(ms)[len(ms) // 2]}

    def spec(t):
        mc = t.memory_config()
        return {"shape": str(t.padded_shape), "dtype": str(t.dtype), "buf": str(mc.buffer_type),
                "layout": str(mc.memory_layout), "addr": t.buffer_address()}

    def probe(q, k, v, mask, **kw):
        if not res:
            res["spec"] = [spec(t) for t in (q, k, v)]
            res["ckc"] = str(kw.get("ckc"))
            arms = {"block": (q, k, v)}
            arms["clone"] = tuple(ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v))
            arms["host"] = tuple(ttnn.from_torch(ttnn.to_torch(t), dtype=t.dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v))
            for name, (q2, k2, v2) in arms.items():
                res[name] = timed(q2, k2, v2, mask, kw)
                res[name + "_hifi2"] = timed(q2, k2, v2, mask, {**kw, "ckc": None})
            for name in ("clone", "host"):
                for t in arms[name]:
                    ttnn.deallocate(t)
        return orig(q, k, v, mask, **kw)

    S.lm_sdpa = probe
    S.embed_sequences(model, seqs, batch_size=8)
    S.lm_sdpa = orig
    res.update(model=a.model, fast=a.fast, calls=a.calls)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Device time of the Boltz-2 token DiT's head re-assembly, [1, 16, 512, 64] -> [1, 512, 768].

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/headcat.py --out headcat.jsonl

cen14 (Boltz-2 --fast, Wormhole, 1000 MHz, main 38dbea510): the attention output is sliced to the 48-lane
head dim, permuted, reshaped and permuted back, 4800 times a fold. The reshape [1, 16, 48, 512] ->
[1, 768, 512] alone reads 40.9 us (0.20 s/fold): 48 is not a tile multiple, so merging heads repacks every
tile. The four ops together are 0.34 s/fold. Each arm here produces the same tensor another way; pure data
movement, so an arm counts only if it is bit-equal to torch.

Device time is a captured trace replayed back to back (op_probe.dev_time), AICLK read during the replays.
"""
import argparse, json, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import aiclk, dev_time, opened_nodes  # noqa: E402

H, S, DP, D = 16, 512, 64, 48


def model(o):
    o = o[:, :, :, :D]
    o = ttnn.permute(o, (0, 1, 3, 2))
    o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
    return ttnn.permute(o, (0, 2, 1))


def concat_heads(o):
    return ttnn.transformer.concatenate_heads(o[:, :, :, :D])


def p0213(o):
    o = ttnn.permute(o, (0, 2, 1, 3))
    o = o[:, :, :, :D]
    return ttnn.reshape(o, (1, S, H * D))


def row_major(o):
    o = ttnn.to_layout(o[:, :, :, :D], ttnn.ROW_MAJOR_LAYOUT)
    o = ttnn.permute(o, (0, 2, 1, 3))
    o = ttnn.reshape(o, (1, S, H * D))
    return ttnn.to_layout(o, ttnn.TILE_LAYOUT)


ARMS = {"model": model, "concat_heads": concat_heads, "p0213": p0213, "row_major": row_major}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = T.get_device()
    nodes = opened_nodes()
    log = a.out.open("a")

    def emit(**kw):
        kw.setdefault("aiclk", aiclk(nodes)); kw.update(arch=T.arch_name(), t=time.time())
        log.write(json.dumps(kw) + "\n"); log.flush(); print(json.dumps(kw), flush=True)

    torch.manual_seed(0)
    x = torch.randn(1, H, S, DP).to(torch.bfloat16)
    ref = x[:, :, :, :D].permute(0, 2, 1, 3).reshape(1, S, H * D)
    o = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for name, f in ARMS.items():
        try:
            y = f(o)
            got = ttnn.to_torch(y)
            ttnn.deallocate(y)
            exact = tuple(got.shape) == tuple(ref.shape) and torch.equal(got, ref)
            us, sp, ck = dev_time(dev, lambda: f(o), nodes=nodes)
            emit(ev="arm", arm=name, us=round(us, 1), spread=round(sp, 3), exact=exact, aiclk=ck)
        except Exception as e:  # an arm ttnn refuses is a result, not a crash
            emit(ev="arm", arm=name, error=f"{type(e).__name__}: {str(e)[:300]}")
    emit(ev="end")


if __name__ == "__main__":
    main()

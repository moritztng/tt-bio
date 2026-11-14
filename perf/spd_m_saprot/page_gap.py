"""Why the gapboard's SaProt page cell reads slower than esmc_ab's batch7_ms on the same tree.

esmc_ab times embed_sequences back to back after a device sync. The gapboard (perf/wh-embed/embed_split_screen.py)
spends ~1.4 s writing npz files between forwards, so the device sits idle before every timed call. One process, one
model load, the page batch (7 x the fixture), arms interleaved for --rounds:
  b2b:   back to back, as esmc_ab
  sleep: --gap seconds of time.sleep before each call (device and host idle)
  busy:  --gap seconds of host CPU work before each call (what the npz write does)
Each call is also split into host prep (tokenize + pad), model forward (dispatch .. readback) and host post.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/page_gap.py --model saprot-35m --fast --out gap.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-35m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--gap", type=float, default=1.4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import saprot as S

    model = S.load_saprot(a.model, fast=a.fast)
    page = Path(a.seq_file).read_text().strip()
    batch = {f"s{i}": page for i in range(7)}
    split = {}
    cls = type(model)
    orig = cls.__call__

    def timed_call(self, *args, **kw):
        t0 = time.perf_counter()
        r = orig(self, *args, **kw)
        split["fwd"] = time.perf_counter() - t0
        split["fwd_end"] = time.perf_counter()
        return r

    cls.__call__ = timed_call

    def busy(sec):
        x = np.random.default_rng(0).standard_normal(1 << 20).astype(np.float32)
        t = time.perf_counter() + sec
        while time.perf_counter() < t:
            np.sort(x)

    nodes = [n for n in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if n]

    def aiclk():   # MHz of the pinned node(s) right now; the harness samples only once a second
        v = []
        for n in nodes:
            try:
                v.append(int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0]))
            except (OSError, ValueError, IndexError):
                pass
        return min(v) if v else None

    gaps = {"b2b": lambda: None, "sleep": lambda: time.sleep(a.gap), "busy": lambda: busy(a.gap)}
    for _ in range(2):
        S.embed_sequences(model, batch, batch_size=8)
    out = open(a.out, "w")
    for rnd in range(a.rounds):
        for arm, gap in gaps.items():
            tot, fwd, pre, post, mhz0, mhz1 = [], [], [], [], [], []
            for _ in range(a.repeat):
                gap()
                mhz0.append(aiclk())
                t0 = time.perf_counter()
                S.embed_sequences(model, batch, batch_size=8)
                t1 = time.perf_counter()
                mhz1.append(aiclk())
                tot.append(t1 - t0)
                fwd.append(split["fwd"])
                post.append(t1 - split["fwd_end"])
                pre.append(tot[-1] - fwd[-1] - post[-1])
            ms = lambda v: [round(x * 1000, 2) for x in v]
            rec = dict(model=a.model, fast=a.fast, round=rnd, arm=arm, gap_s=a.gap, total_ms=ms(tot), fwd_ms=ms(fwd),
                       pre_ms=ms(pre), post_ms=ms(post), mhz_before=mhz0, mhz_after=mhz1, median_ms=round(sorted(tot)[len(tot) // 2] * 1000, 2))
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

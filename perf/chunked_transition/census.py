#!/usr/bin/env python3
"""E6 census: which Transition shapes the fold actually runs, how many times, and what they cost.

Counts every `_transition_linear` call by (mt, kt, nt, fired) inside a real 298 aa fold, then
crosses the counts with the ladder measured on this card. Answers the only question the fold A/B
leaves open: is the band worth what the op-isolated ladder says it is?
"""
import json
import sys
import time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/perfwar-chunked-transition-cb")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "gpu_vs_tt"))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "protenix-v2"
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    import ttnn
    from tt_bio import tenstorrent as T

    counts = {}
    orig = T._transition_linear
    dev_t = {"n": 0}

    def counting(x, w, ckc, dtype, memory_config, activation=None):
        xs, ws = list(x.padded_shape), list(w.padded_shape)
        mt = 1
        for d in xs[:-2]:
            mt *= int(d)
        mt *= -(-int(xs[-2]) // 32)
        kt, nt = -(-int(xs[-1]) // 32), -(-int(ws[-1]) // 32)
        y = orig(x, w, ckc, dtype, memory_config, activation=activation)
        cfg = None
        try:
            mv = ttnn.get_memory_view(x.device(), ttnn.BufferType.L1)
            ob = 0
            if memory_config.buffer_type == ttnn.BufferType.L1:
                ob = -(-(mt * nt) // mv.num_banks) * 2048
            cfg = T._transition_program_config(
                mt, kt, nt, 2,
                (mv.largest_contiguous_bytes_free_per_bank - ob) // 65536 * 65536,
                activation == "silu")
        except Exception:
            pass
        k = (mt, kt, nt, bool(cfg), getattr(cfg, "in0_block_w", 0), getattr(cfg, "out_block_h", 0))
        counts[k] = counts.get(k, 0) + 1
        dev_t["n"] += 1
        return y

    T._transition_linear = counting

    import tt_baseline as B
    B.RECYCLING_STEPS = cycles
    B.SAMPLING_STEPS = 8
    one_fold, meta, state = B.build_fold(
        model, Path("/tmp/e6-msa"), WT / "examples" / "prot300.yaml",
        WT / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m")
    counts.clear()
    t0 = time.perf_counter()
    _s, m = one_fold()
    print(f"fold {time.perf_counter() - t0:.1f}s plddt {m.get('plddt')} "
          f"calls {sum(counts.values())} distinct {len(counts)}")
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"{'mt':>6} {'kt':>4} {'nt':>4} {'fired':>6} {'bw':>3} {'obh':>4} {'calls':>7}")
    for (mt, kt, nt, fired, bw, obh), n in rows:
        print(f"{mt:>6} {kt:>4} {nt:>4} {fired!s:>6} {bw:>3} {obh:>4} {n:>7}")
    out = WT / "perf" / "chunked_transition" / f"census_{model}.json"
    out.write_text(json.dumps([dict(mt=k[0], kt=k[1], nt=k[2], fired=k[3], bw=k[4], obh=k[5], n=v)
                               for k, v in rows], indent=1) + "\n")
    state.reset()
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())

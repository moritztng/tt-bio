"""Per-site cost of the pairformer block's deletion candidates, measured directly on Blackhole.

`perf/util_op_deletes/sites.py` priced these by aligning a WH graph capture onto the BH kernel
census by op CLASS. That alignment is ambiguous inside a run of same-class ops and it got one
wrong: it charged 0.7292 ms/block to `ttnn.reshape(x, (1, *x.shape))`, which this file's companion
`reshape_probe.py` shows returns the input buffer unchanged in 0.006 ms. So the per-row numbers in
that table cannot pick the next deletion.

This asks the device instead. Every shape, dtype and memory space below is read off the BH op
ledger of one settled 512 aa PairformerLayer (`perf/util_op_deletes/ledger_block_512_qb2c3_
patched.json`), so each row is the op production actually issues, and the number is what deleting
that op would hand back at the kernel.
"""
import json
import statistics as st
import time

import torch

torch.set_grad_enabled(False)
import ttnn
import tt_bio.tenstorrent as T

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
REPS = 5
dev = T.get_device()


def mk(shape, mc):
    return ttnn.from_torch(torch.randn(*shape, dtype=torch.float32).bfloat16(),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


def bench(label, calls_per_block, setup, run):
    """`setup` returns the operand tuple; `run` issues the op once. Median of REPS after a warm."""
    try:
        args = setup()
    except Exception as e:                                                      # noqa: BLE001
        print("%-46s ALLOC REFUSED %s" % (label, str(e)[:70]))
        return None
    ts = []
    try:
        for i in range(REPS + 1):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = run(*args)
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) * 1e3
            if i:
                ts.append(dt)
            if out is not None and out.buffer_address() not in {a.buffer_address() for a in args}:
                ttnn.deallocate(out)
    except Exception as e:                                                      # noqa: BLE001
        print("%-46s RUN REFUSED %s" % (label, str(e)[:70]))
        return None
    finally:
        for a in args:
            try:
                ttnn.deallocate(a)
            except Exception:                                                   # noqa: BLE001
                pass
    ms = st.median(ts)
    row = {"site": label, "n_per_block": calls_per_block, "ms_per_call": ms,
           "ms_per_block": ms * calls_per_block, "s_per_fold": ms * calls_per_block * 280 / 1e3,
           "samples_ms": ts}
    print("%-46s %2d x %8.4f ms = %8.4f ms/block  %7.4f s/fold" % (
        label, calls_per_block, ms, row["ms_per_block"], row["s_per_fold"]))
    return row


P = (1, 512, 512, 128)      # the pair tensor, 67.1 MB bf16
CH = (1, 128, 512, 512)     # the same bytes in the trimul's channel-major layout
HM = (512, 4, 512, 32)      # the triangle attention's head-major activation

rows = []
print("per-call medians of %d, warm, qb2 p300c card 1, 512 aa shapes\n" % REPS)

rows.append(bench("trimul pair mask  mul_(CH DRAM, 1x1x512x512 DRAM)", 2,
                  lambda: (mk(CH, DRAM), mk((1, 1, 512, 512), DRAM)),
                  lambda a, b: ttnn.multiply_(a, b)))

rows.append(bench("trimul out gate   mul_(P L1, P DRAM) sigmoid", 2,
                  lambda: (mk(P, L1), mk(P, DRAM)),
                  lambda a, b: ttnn.multiply_(
                      a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])))

rows.append(bench("triatt out gate   mul_(HM DRAM, HM DRAM) sigmoid", 2,
                  lambda: (mk(HM, DRAM), mk(HM, DRAM)),
                  lambda a, b: ttnn.multiply_(
                      a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])))

rows.append(bench("residual          add_(P DRAM, P L1)", 3,
                  lambda: (mk(P, DRAM), mk(P, L1)),
                  lambda a, b: ttnn.add_(a, b)))

rows.append(bench("residual          add_(P DRAM, P DRAM)", 2,
                  lambda: (mk(P, DRAM), mk(P, DRAM)),
                  lambda a, b: ttnn.add_(a, b)))

rows.append(bench("transition slice  P DRAM -> 1x16x512x128 x32", 32,
                  lambda: (mk(P, DRAM),),
                  lambda a: ttnn.slice(a, (0, 0, 0, 0), (1, 16, 512, 128))))

rows.append(bench("pair transpose    permute(1,0,2) 512x512x128 DRAM->L1", 2,
                  lambda: (mk((512, 512, 128), DRAM),),
                  lambda a: ttnn.permute(a, (1, 0, 2), memory_config=L1)))

rows = [r for r in rows if r]
tot = sum(r["s_per_fold"] for r in rows)
print("\nmeasured candidates total %.4f s/fold" % tot)
json.dump({"arch": "BH p300c qb2 card 1", "tokens": 512, "blocks_per_fold": 280, "rows": rows},
          open("/home/ttuser/scratch/uod2/site_bench.json", "w"), indent=1)

#!/usr/bin/env python3
"""bcx-p10-devtop: does rne_add serve the taped residual on cell E? One block step, counters out."""
import json, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack import stack as S
from perf.bcx_p10_shape import shape as SH
from perf.bcx_p10_mmroof import cells as CE


class A:
    params = None; card = 0; seed = 0; threads = 8


def main():
    from perf.bcx_afgrad import afgrad as AF
    args = A(); args.params = AF.DEFAULT_PARAMS
    print("ARMED", json.dumps(CE.arm(1, 1, 1, 1)), flush=True)
    lv, dev, ref = S.open_all(args)
    spec = SH.CELLS["E"]; n32 = spec["pad"]
    raw_m, raw_z, _, _ = S.inputs(ref, SH.N_HOST, 0, ragged=True)
    m0, z0 = SH.pad_like_fold(raw_m, raw_z, n32)
    m0 = m0.repeat(spec["depth"], 1, 1)
    wm = torch.randn(m0.shape) / m0.numel() ** 0.5
    wz = torch.randn(z0.shape) / z0.numel() ** 0.5
    masks = SH.cell_masks(dev, n32, SH.N_REAL, spec["depth"], spec["masks"])
    SH.set_hifi(True)
    from tt_bio import rne_add, taped_ttnn, af2
    for stack in ("evo", "extra"):
        S.block_step(dev, lv, m0, z0, wm, wz, stack, k=1, ckpt=spec["ckpt"], masks=masks)
        print(stack, json.dumps({"stats": rne_add.STATS, "enabled": rne_add.enabled(),
              "cls_rne_kernel": af2.AF2PairBlock.rne_kernel,
              "entry": list(taped_ttnn.KERNEL_STATS.get("rne_add", [])) if hasattr(taped_ttnn, "KERNEL_STATS") else None,
              "rejects": {repr(k): v for k, v in rne_add.REJECTS.items()}}), flush=True)


if __name__ == "__main__":
    main()

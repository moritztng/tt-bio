#!/usr/bin/env python3
"""Which of the two conditioning sites costs the accuracy? One CIF per arm, scored separately.

The stacked lever moves cdk2x2_512 by 12.56 A against a measured seed-scatter band of
5.35-9.90 A on the same fixture and metric, and drops mean plDDT to 84.47 against a seed band of
85.32-86.51. The two sites are not the same kind of change, so they cannot be judged together:

  adaln   AdaLN: moves the gate's sigmoid into the s_scale matmul epilogue AND folds the pair.
          The epilogue move is bit-identical to sigmoid(linear(bias=b)) in isolation, but it
          also changes WHERE the gate is rounded relative to the chain.
  attn    DiffusionTransformerLayer write-back: a pure addcmul substitution. `s_o` already
          carries activation="sigmoid", so nothing about its producer changes.

Untimed on purpose -- the box is shared and this run answers an accuracy question only.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--size", default="512")
    ap.add_argument("--out", type=Path, default=ROOT / "perf/c12_eltwise/runs/acc_arms")
    a = ap.parse_args()

    import tt_baseline as B
    from tt_bio import eltwise_fusion as EF
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        import fold_ab_multi as _FAM
        _FAM.patch_boltz2_cfg()

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        a.model, ROOT / (".msa_s512_%s_%s" % (a.model, a.size)),
        fixdir / ("cdk2x2_%s.yaml" % a.size), fixdir / ("cdk2x2_%s.a3m" % a.size))
    struct_dir = Path(meta["struct_dir"])

    EF.FUSE_COND_MULADD = EF.FUSE_ATTN_GATE_ADD = False
    one_fold()   # cold

    a.out.mkdir(parents=True, exist_ok=True)
    for name, cond, attn in (("off", False, False), ("adaln", True, False),
                             ("attn", False, True), ("both", True, True)):
        EF.FUSE_COND_MULADD, EF.FUSE_ATTN_GATE_ADD = cond, attn
        t, _m = one_fold()
        d = a.out / name
        d.mkdir(exist_ok=True)
        for c in sorted(struct_dir.glob("*.cif")):
            (d / c.name).write_bytes(c.read_bytes())
        print("arm %-6s %8.3f s -> %s" % (name, t, d), flush=True)
    EF.FUSE_COND_MULADD = EF.FUSE_ATTN_GATE_ADD = False


if __name__ == "__main__":
    main()

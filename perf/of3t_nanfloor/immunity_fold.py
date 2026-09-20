#!/usr/bin/env python3
"""D3: one OpenDDE fold at a token count that is not a bucket multiple, with and without the floor.

`opendde.refiner` is the one shipped-ON accurate-softmax site whose input can carry a fully masked
row, so it is where the floor could have a blast radius in a production model. The test is not
"is it finite" -- it is whether the floor changed a single bit of a shipped model's output.

Four arms in ONE process, one device context, same weights, same MSA:

    floored    the tree as it stands
    unfloored  `_accurate_softmax` swapped for a verbatim copy with the clamp line removed
    floored2   the A/A control -- without it a matching digest proves nothing, because an
               unseeded diffusion sampler would not reproduce its own bytes either
    probe      floored, plus a per-call count of how often the clamp actually BINDS and how
               far `ttnn.max` overshoots the true row maximum. A floor that never binds is a
               different claim from a floor that binds and changes nothing.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

PROBE = {"calls": 0, "rows": 0, "bound": 0, "rows_floor_binds": 0,
         "rows_overshot": 0, "rows_fully_masked": 0, "worst_row_overshoot": -1e30,
         "worst_dmin": 0.0, "nonfinite_out": 0}


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opendde")
    ap.add_argument("--size", type=int, default=385)
    ap.add_argument("--arms", default="floored,unfloored,floored2,probe")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import os
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)
    import tt_baseline as B

    shipped_fn = T._accurate_softmax

    def unfloored(x, compute_kernel_config=None, fp32: bool = True):
        """`_accurate_softmax` verbatim with the `ttnn.clamp(d, -60.0, None)` line removed."""
        src_dtype = x.dtype
        xf = (ttnn.typecast(x, ttnn.float32, memory_config=x.memory_config())
              if fp32 and src_dtype != ttnn.float32 else x)
        m = ttnn.max(xf, dim=-1, keepdim=True)
        d = ttnn.subtract(xf, m)
        ttnn.deallocate(m)
        if xf is not x:
            ttnn.deallocate(xf)
        ttnn.exp(d, output_tensor=d)
        s = ttnn.sum(d, dim=-1, keepdim=True, compute_kernel_config=compute_kernel_config)
        p = ttnn.divide(d, s)
        ttnn.deallocate(d)
        ttnn.deallocate(s)
        if p.dtype != src_dtype:
            q = ttnn.typecast(p, src_dtype, memory_config=p.memory_config())
            ttnn.deallocate(p)
            p = q
        return p

    def probed(x, compute_kernel_config=None, fp32: bool = True):
        """The shipped function, with the quantity the floor exists for measured on every call."""
        xf = (ttnn.typecast(x, ttnn.float32, memory_config=x.memory_config())
              if fp32 and x.dtype != ttnn.float32 else x)
        m = ttnn.max(xf, dim=-1, keepdim=True)
        d = ttnn.subtract(xf, m)
        # PER ROW. A row whose max ttnn.max read too high has max_j d_ij < 0, and a single such
        # row is invisible in a global maximum over the tensor because every healthy row pins it
        # to 0. So reduce along the softmax axis and take the WORST row.
        rowmax = ttnn.to_torch(ttnn.max(d, dim=-1, keepdim=True)).float()
        rowmin = ttnn.to_torch(ttnn.min(d, dim=-1, keepdim=True)).float()
        m_h = ttnn.to_torch(m).float()
        PROBE["calls"] += 1
        PROBE["rows"] += int(rowmax.numel())
        PROBE["rows_overshot"] += int((rowmax < 0).sum())
        PROBE["worst_row_overshoot"] = max(PROBE["worst_row_overshoot"],
                                           float(-rowmax.min()))
        PROBE["rows_fully_masked"] += int((m_h < -1e6).sum())
        PROBE["bound"] += int(float(rowmin.min()) < -60.0)
        PROBE["rows_floor_binds"] += int((rowmin < -60.0).sum())
        PROBE["worst_dmin"] = min(PROBE["worst_dmin"], float(rowmin.min()))
        ttnn.deallocate(m)
        ttnn.deallocate(d)
        if xf is not x:
            ttnn.deallocate(xf)
        y = shipped_fn(x, compute_kernel_config, fp32)
        PROBE["nonfinite_out"] += int(not bool(torch.isfinite(ttnn.to_torch(y)).all()))
        return y

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    assert tgt.is_file() and a3m.is_file(), f"no fixture at {tgt}"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_nf_{a.model}_{a.size}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])

    res = {"model": a.model, "size": a.size,
           "site_default_opendde_refiner": T.accurate_softmax_site("opendde.refiner",
                                                                   default=True),
           "accurate_softmax_ab_env": os.environ.get("TT_BIO_ACCURATE_SOFTMAX_AB"),
           "bucket_multiple_of_32": a.size % 32 == 0, "runs": []}
    print("=== cold fold ===", flush=True)
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.1f}s n_tokens={cold_m.get('n_tokens')} plddt={cold_m.get('plddt')}",
          flush=True)
    res["n_tokens"] = cold_m.get("n_tokens")

    for arm in a.arms.split(","):
        T._accurate_softmax = {"floored": shipped_fn, "floored2": shipped_fn,
                               "unfloored": unfloored, "probe": probed}[arm]
        for k in PROBE:
            PROBE[k] = {"worst_dmin": 0.0, "worst_row_overshoot": -1e30}.get(k, 0)
        t0 = time.perf_counter()
        try:
            fold_s, m = one_fold()
            err = None
        except Exception as e:                                                  # noqa: BLE001
            fold_s, m, err = time.perf_counter() - t0, {}, f"{type(e).__name__}: {e}"[:400]
        rec = {"arm": arm, "fold_s": round(fold_s, 2), "error": err,
               "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
               "sha256": (sha_dir(struct_dir) if err is None else None),
               "probe": (dict(PROBE) if arm == "probe" else None)}
        keep = a.out.parent / f"{a.out.stem}_struct" / arm
        keep.mkdir(parents=True, exist_ok=True)
        for p in struct_dir.glob("*"):
            if p.is_file():
                (keep / p.name).write_bytes(p.read_bytes())
        res["runs"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm} {fold_s:.1f}s plddt={m.get('plddt')} err={err}", flush=True)
        print(f"    sha {json.dumps(rec['sha256'])}", flush=True)
        if arm == "probe":
            print(f"    probe {json.dumps(rec['probe'])}", flush=True)

    T._accurate_softmax = shipped_fn
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

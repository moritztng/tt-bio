#!/usr/bin/env python3
"""Upstream 0.4.3 msa_module forward, site by site, out of one process.

The module is driven through its own submodules in the order `MSAModuleBlock.forward` and
`PairBlock.forward` call them, with every residual add explicit, so each site's input state and
update can be recorded and each site can be re-run alone. `--policy f64 --trace-out` writes the
float64 trace every other arm is scored against; that arm must reproduce the capture's z_out.

  chain  the whole module in the policy, recording every state and update (propagated error)
  tf     teacher-forced: each site fed the float64 trace's input state, its update compared with
         the float64 update (the site's own local error, nothing propagated)

`--resid-bf16` rounds m and z to bf16 on entry and after every residual add while the ops run in
the policy. That is our device arm's residual storage imposed on upstream's own arithmetic.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_tapeamp"))
from amp_arm import cast, upstream_identity  # noqa: E402


def site_fns(blk, msa_mask, pair_mask):
    """name -> (fn(m, z) -> update, which track the update lands on), in execution order."""
    ps = blk.pair_stack
    f = {"opm": (lambda m, z: blk.outer_product_mean(m, mask=msa_mask, chunk_size=None,
                                                      inplace_safe=False), "z")}
    if not blk.skip_msa_update:
        f["pwa"] = (lambda m, z: blk.msa_att_row(m, z=z, mask=pair_mask, chunk_size=None), "m")
        f["msa_transition"] = (lambda m, z: blk.msa_transition(
            m, mask=msa_mask, chunk_size=None, ckpt_chunk_size=None), "m")
    f["tri_mul_out"] = (lambda m, z: ps.tri_mul_out(z, mask=pair_mask, inplace_safe=False,
                                                     _add_with_inplace=True), "z")
    f["tri_mul_in"] = (lambda m, z: ps.tri_mul_in(z, mask=pair_mask, inplace_safe=False,
                                                   _add_with_inplace=True), "z")
    f["tri_att_start"] = (lambda m, z: ps.tri_att_start(z, mask=pair_mask, chunk_size=None), "z")
    # PairBlock runs the ending node on the transposed pair and transposes back; adding the
    # transposed update is the same elementwise sum.
    f["tri_att_end"] = (lambda m, z: ps.tri_att_end(
        z.transpose(-2, -3), mask=pair_mask.transpose(-1, -2),
        chunk_size=None).transpose(-2, -3), "z")
    f["pair_transition"] = (lambda m, z: ps.pair_transition(z, mask=pair_mask, chunk_size=None),
                            "z")
    return f


def score_sites(rec, T, n, tokm, chain, dtypes, z_out=None):
    """Every site against the float64 trace T on the real token block. Shared with ours_fwd.py
    so both sides are scored by one function. With T None the record scores itself (the f64
    reference arm, which must read 0.0 everywhere)."""
    real_z = lambda x: x.reshape(n, n, -1)[tokm][:, tokm]
    real_m = lambda x: x.reshape(x.shape[-3], n, -1)[:, tokm]
    nrm = lambda x: float(x.norm())
    ref = T["sites"] if T else rec
    zo = nrm(real_z(T["z_out"] if T else z_out))
    rows = {}
    for key, r in rec.items():
        t = ref[key]
        trk = "m" if key.split(".")[1] in ("pwa", "msa_transition") else "z"
        R = real_m if trk == "m" else real_z
        du = R(r["upd"]) - R(t["upd"])
        row = {"track": trk, "upd_ref_norm": nrm(R(t["upd"])), "upd_err_norm": nrm(du),
               "upd_rel": nrm(du) / nrm(R(t["upd"])),
               "upd_err_over_zout": nrm(du) / zo if trk == "z" else None,
               "upd_err_over_m_state": nrm(du) / nrm(real_m(t["m"])) if trk == "m" else None,
               "dtypes": dtypes.get(key)}
        if chain and T:
            row["z_in_rel"] = nrm(real_z(r["z"]) - real_z(t["z"])) / nrm(real_z(t["z"]))
            row["m_in_rel"] = nrm(real_m(r["m"]) - real_m(t["m"])) / nrm(real_m(t["m"]))
        rows[key] = row
    return rows


def print_rows(rows):
    for k, r in rows.items():
        e = r["upd_err_over_zout"]
        print(f"  {k:22s} upd_rel {r['upd_rel']:.4e}  err/|z_out| "
              f"{e if e is not None else float('nan'):.4e}"
              + (f"  z_in_rel {r['z_in_rel']:.4e}" if "z_in_rel" in r else ""), flush=True)


def load_msa(dt, ckpt):
    """Upstream 0.4.3's msa_module at dtype dt with the checkpoint's weights (D141 guard)."""
    import bundle_min as BM
    model = BM.build(dt, 20260919, "cpu", num_recycles=0)[1]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items() if k.startswith("msa_module.")}
    got = model.load_state_dict(sd, strict=False)
    miss = [k for k in got.missing_keys if k.startswith("msa_module.")]
    if miss or got.unexpected_keys:
        print(f"HARD FAILURE: {len(miss)} missing, {len(got.unexpected_keys)} unexpected")
        return None
    return model.msa_module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=("bf16auto", "f32", "f64"))
    ap.add_argument("--mode", required=True, choices=("chain", "tf"))
    ap.add_argument("--boundary", type=Path, required=True)
    ap.add_argument("--trace", type=Path, help="the f64 trace (required unless writing it)")
    ap.add_argument("--trace-out", type=Path, dest="trace_out")
    ap.add_argument("--resid-bf16", action="store_true", dest="resid_bf16")
    ap.add_argument("--ckpt", type=Path, default=Path("/home/ttuser/of3-weights/of3-p2-155k.pt"))
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    root, ver, meta, ln_sig, dig = upstream_identity()
    if ver != "0.4.3" or dig["version_from_digest"] != "0.4.3":
        print(f"HARD FAILURE: resolved {ver}, digest {dig}")
        return 3
    import bundle_min as BM

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    (m0, z0), kw = B["inputs"]["args"], B["inputs"]["kwargs"]
    z_ref = B["outputs"]
    n = int(z_ref.shape[-2])
    tokm = torch.diagonal(kw["pair_mask"].reshape(n, n)) > 0

    dt = torch.float64 if a.policy == "f64" else torch.float32
    mm = load_msa(dt, a.ckpt)
    if mm is None:
        return 5

    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else BM.no_autocast() if a.policy == "f64" else torch.autocast("cpu", enabled=False))
    msa_mask, pair_mask = cast(kw["msa_mask"], dt), cast(kw["pair_mask"], dt)
    rnd = ((lambda x: x.to(torch.bfloat16).to(x.dtype)) if a.resid_bf16 else (lambda x: x))
    d64 = lambda x: x.detach().to(torch.float64).clone()
    T = torch.load(a.trace, map_location="cpu", weights_only=False) if a.trace else None

    rec, dtypes = {}, {}
    with torch.no_grad(), ctx:
        if a.mode == "chain":
            m, z = rnd(cast(m0, dt)), rnd(cast(z0, dt))
            for i, blk in enumerate(mm.blocks):
                for name, (fn, trk) in site_fns(blk, msa_mask, pair_mask).items():
                    upd = fn(m, z)
                    rec[f"b{i}.{name}"] = {"m": d64(m), "z": d64(z), "upd": d64(upd)}
                    if trk == "z":
                        z = rnd(z + upd)
                    else:
                        m = rnd(m + upd)
                    dtypes[f"b{i}.{name}"] = {"update": str(upd.dtype), "state": str(z.dtype)}
            z_out = d64(z)
        else:
            for i, blk in enumerate(mm.blocks):
                for name, (fn, trk) in site_fns(blk, msa_mask, pair_mask).items():
                    t = T["sites"][f"b{i}.{name}"]
                    upd = fn(cast(t["m"], dt), cast(t["z"], dt))
                    rec[f"b{i}.{name}"] = {"upd": d64(upd)}
                    dtypes[f"b{i}.{name}"] = {"update": str(upd.dtype)}
            z_out = None

    rows = score_sites(rec, T, n, tokm, a.mode == "chain", dtypes, z_out)
    out = {"policy": a.policy, "mode": a.mode, "resid_bf16": a.resid_bf16,
           "boundary": str(a.boundary), "trace": str(a.trace) if a.trace else None,
           "upstream_version": ver, "upstream_pkg": root, "host": socket.gethostname(),
           "torch": torch.__version__, "n_tokens": n, "n_real": int(tokm.sum()), "sites": rows}
    if z_out is not None:
        zr = z_ref.double().reshape(n, n, -1)[tokm][:, tokm]
        zo = z_out.reshape(n, n, -1)[tokm][:, tokm]
        out["z_out_vs_capture_rel"] = float((zo - zr).norm() / zr.norm())
        out["z_out_ref_norm"] = float(zr.norm())
        print(f"z_out vs capture, real block: {out['z_out_vs_capture_rel']:.6e}", flush=True)
    if a.trace_out:
        if a.policy != "f64" or a.mode != "chain" or a.resid_bf16:
            print("HARD FAILURE: only the plain f64 chain may write the reference trace")
            return 7
        if out["z_out_vs_capture_rel"] > 1e-12:
            print("HARD FAILURE: the driver does not reproduce the capture")
            return 8
        torch.save({"sites": rec, "z_out": z_out, "boundary": str(a.boundary)}, a.trace_out)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(out, indent=1) + "\n")
    print_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

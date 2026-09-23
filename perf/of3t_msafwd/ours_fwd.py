#!/usr/bin/env python3
"""Our device msa_module forward, site by site, against the float64 trace up_fwd.py wrote.

The block is driven through the same device primitives, in the same order and with the same
arguments, as `MSAModuleBlock.__call__`. `--mode dev` keeps every residual on device exactly as
the shipped block does (bf16 `ttnn.add_`) and must reproduce `MSAModule.__call__`'s z_out bit for
bit before any site number is read. The other modes change one thing each:

  tf      teacher-forced: each site fed the float64 trace's input state (uploaded bf16, as every
          device op receives it), its update compared with the float64 update
  host    residual stream held on the HOST at `--resid` precision (fp32 or bf16); each op is
          still fed the bf16 upload of the current state and its bf16 update is added on host.
          fp32 is upstream's own residual storage imposed on our ops; bf16 must land on `dev`.
  dev32   residual stream held on device in fp32 (typecast in, bf16 copy out to each op): the
          shape a device fix would take.

Inference ops only: no tape. of3t-msaamp's number came through `ops.checkpoint_segment`, whose
forward runs under `no_grad` and therefore takes the shipped op at every site; `--mode dev` is
compared against that banked value as well.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("dev", "tf", "host", "dev32"))
    ap.add_argument("--resid", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--tracks", default="zm", choices=("zm", "z", "m"),
                    help="which residual tracks --mode host/dev32 hold at the wider precision")
    ap.add_argument("--ln-host", action="store_true", dest="ln_host",
                    help="substitution arm: every ttnn.layer_norm computed on the host in float64 "
                         "with the checkpoint's own (unrounded) gamma/beta, output uploaded at "
                         "the input's dtype. Its input is still the bf16 state the op receives.")
    ap.add_argument("--boundary", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device, _RESIDUAL_L1, _residual_update_memory_config
    from tt_bio.openfold3_msa_embedder import MSAModule
    from tt_bio.openfold3_weights import is_openbind
    from up_fwd import score_sites, print_rows

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    (m0, z0), kw = B["inputs"]["args"], B["inputs"]["kwargs"]
    n, n_seq = int(z0.shape[-2]), int(m0.shape[-3])
    T = torch.load(a.trace, map_location="cpu", weights_only=False)
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}

    dev = get_device()
    # The instrument's config (msa_instrument.py), which is what the banked 8.176e-03 ran on.
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    src, census = {}, {}
    orig_ft, orig_ln = ttnn.from_torch, ttnn.layer_norm

    def recording_ft(t, *ar, **k):
        out = orig_ft(t, *ar, **k)
        if torch.is_tensor(t):
            src[id(out)] = (out, t.detach().clone())
        return out

    def caller():
        f = sys._getframe(2)
        while f is not None:
            me = f.f_locals.get("self")
            if me is not None and type(me).__module__.startswith("tt_bio"):
                return type(me).__name__
            f = f.f_back
        return "?"

    def ln_wrap(x, *ar, weight=None, bias=None, epsilon=1e-5, memory_config=None, **k):
        key = f"{caller()}|{x.dtype}"
        census[key] = census.get(key, 0) + 1
        if not a.ln_host:
            return orig_ln(x, *ar, weight=weight, bias=bias, epsilon=epsilon,
                           memory_config=memory_config, **k)
        xs = torch.Tensor(ttnn.to_torch(x)).double()
        c = xs.shape[-1]
        g = src[id(weight)][1].double().reshape(-1)[:c] if weight is not None else 1.0
        b = src[id(bias)][1].double().reshape(-1)[:c] if bias is not None else 0.0
        mu = xs.mean(-1, keepdim=True)
        y = (xs - mu) / (xs.var(-1, unbiased=False, keepdim=True) + epsilon).sqrt() * g + b
        return orig_ft(y.float(), dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                       memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)

    ttnn.from_torch = recording_ft
    try:
        msa = MSAModule(sd, ckc, transpose_bias=not is_openbind(sd))
    finally:
        ttnn.from_torch = orig_ft
    del sd
    ttnn.layer_norm = ln_wrap

    up = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(
        x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    dn = lambda t: torch.Tensor(ttnn.to_torch(t)).double()
    pm = kw["pair_mask"].reshape(1, n, n)
    m1 = torch.diagonal(pm.reshape(n, n)).reshape(1, n).clamp(0, 1)
    pm_d = up(pm)
    attn_d = up((1 - m1).reshape(1, 1, 1, n) * -1e9)
    tokm = m1.reshape(-1).bool()
    m_shape, z_shape = (1, n_seq, n, int(m0.shape[-1])), (1, n, n, int(z0.shape[-1]))

    def site(blk, name, m_d, z_d):
        ps = blk.pair_stack
        if name == "opm":
            u, shp = blk.opm(m_d, None, None), z_shape
        elif name == "pwa":
            u, shp = blk.pwa(m_d, ttnn.clone(z_d), attn_d), m_shape
        elif name == "msa_transition":
            u, shp = blk.msa_transition(m_d), m_shape
        elif name == "tri_mul_out":
            u, shp = ps.triangle_multiplication_start(z_d, pm_d), z_shape
        elif name == "tri_mul_in":
            u, shp = ps.triangle_multiplication_end(z_d, pm_d), z_shape
        elif name == "tri_att_start":
            u, shp = ps.triangle_attention_start(z_d, attn_d), z_shape
        elif name == "tri_att_end":
            u, shp = ps.triangle_attention_end(z_d, attn_d), z_shape
        else:
            u, shp = ps.transition_z(
                z_d, memory_config=_residual_update_memory_config(z_d.shape, z_d.dtype)
                if _RESIDUAL_L1 else None, mask=None), z_shape
        return ttnn.reshape(u, shp)

    def names(blk):
        mid = ("pwa", "msa_transition") if blk.has_msa_update else ()
        return ("opm",) + mid + ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
                                 "pair_transition")

    trk = lambda nm: "m" if nm in ("pwa", "msa_transition") else "z"
    rec, dtypes, z_out = {}, {}, None
    if a.mode == "tf":
        for i, blk in enumerate(msa.blocks):
            for nm in names(blk):
                t = T["sites"][f"b{i}.{nm}"]
                u = site(blk, nm, up(t["m"].reshape(m_shape)), up(t["z"].reshape(z_shape)))
                rec[f"b{i}.{nm}"] = {"upd": dn(u)}
                dtypes[f"b{i}.{nm}"] = {"update": str(u.dtype)}
    elif a.mode == "dev":
        m_d, z_d = up(m0.reshape(m_shape)), up(z0.reshape(z_shape))
        for i, blk in enumerate(msa.blocks):
            for nm in names(blk):
                ms, zs = dn(m_d), dn(z_d)
                u = site(blk, nm, m_d, z_d)
                rec[f"b{i}.{nm}"] = {"m": ms, "z": zs, "upd": dn(u)}
                dtypes[f"b{i}.{nm}"] = {"update": str(u.dtype), "state": str(z_d.dtype)}
                # MSAModuleBlock's own residual forms: opm's sum lands in the update's buffer,
                # the m residuals in the update's (pwa) and in m's (transition), the pair ones in z.
                if nm == "opm":
                    z_d = ttnn.add_(u, z_d)
                elif nm == "pwa":
                    m_d = ttnn.add_(u, m_d)
                elif nm == "msa_transition":
                    m_d = ttnn.add_(m_d, u)
                else:
                    z_d = ttnn.add_(z_d, u)
        z_out = dn(z_d)
        # The same weights through the module's own __call__, on fresh uploads.
        _, z_mod = msa(up(m0.reshape(m_shape)), up(z0.reshape(z_shape)), pair_mask=pm_d,
                       attn_mask=attn_d)
        z_mod = dn(z_mod)
        bitexact = bool(torch.equal(z_mod, z_out))
        print(f"driver vs MSAModule.__call__: bit-identical {bitexact}, "
              f"max|d| {float((z_mod - z_out).abs().max()):.3e}", flush=True)
        if not bitexact:
            print("HARD FAILURE: the driver is not the module")
            return 8
    else:
        wide = {t: (t in a.tracks) and a.resid == "fp32" for t in "zm"}
        if a.mode == "host":
            st = {"m": m0.reshape(m_shape).float(), "z": z0.reshape(z_shape).float()}
            rnd = lambda t, x: x if wide[t] else x.to(torch.bfloat16).float()
            st = {t: rnd(t, x) for t, x in st.items()}
            for i, blk in enumerate(msa.blocks):
                for nm in names(blk):
                    u = site(blk, nm, up(st["m"]), up(st["z"]))
                    ud = dn(u)
                    rec[f"b{i}.{nm}"] = {"m": st["m"].double(), "z": st["z"].double(), "upd": ud}
                    dtypes[f"b{i}.{nm}"] = {"update": str(u.dtype),
                                            "state": "fp32" if wide[trk(nm)] else "bf16"}
                    st[trk(nm)] = rnd(trk(nm), st[trk(nm)] + ud.float())
            z_out = st["z"].double()
        else:  # dev32
            f32, b16 = ttnn.float32, ttnn.bfloat16
            st = {"m": up(m0.reshape(m_shape), f32 if wide["m"] else b16),
                  "z": up(z0.reshape(z_shape), f32 if wide["z"] else b16)}
            narrow = lambda x: ttnn.typecast(x, b16) if x.dtype == f32 else x
            for i, blk in enumerate(msa.blocks):
                for nm in names(blk):
                    ms, zs = dn(st["m"]), dn(st["z"])
                    u = site(blk, nm, narrow(st["m"]), narrow(st["z"]))
                    rec[f"b{i}.{nm}"] = {"m": ms, "z": zs, "upd": dn(u)}
                    t = trk(nm)
                    dtypes[f"b{i}.{nm}"] = {"update": str(u.dtype), "state": str(st[t].dtype)}
                    if st[t].dtype == f32:
                        st[t] = ttnn.add(st[t], ttnn.typecast(u, f32))
                    else:
                        st[t] = ttnn.add_(st[t], u)
            z_out = dn(st["z"])

    rows = score_sites(rec, T, n, tokm, a.mode != "tf", dtypes, z_out)
    out = {"mode": a.mode, "resid": a.resid, "tracks": a.tracks, "boundary": str(a.boundary),
           "trace": str(a.trace), "host": socket.gethostname(),
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "board": open(f"/sys/class/tenstorrent/tenstorrent!{os.environ.get('TT_VISIBLE_DEVICES', '0')}/tt_card_type").read().strip(),
           "compute_kernel_config": "HiFi4 fp32_dest_acc packer_l1_acc (msa_instrument.py's)",
           "residual_l1": bool(_RESIDUAL_L1), "ln_host": a.ln_host, "ln_census": census, "n_tokens": n, "n_real": int(tokm.sum()),
           "sites": rows}
    if z_out is not None:
        zr = B["outputs"].double().reshape(n, n, -1)[tokm][:, tokm]
        zo = z_out.reshape(n, n, -1)[tokm][:, tokm]
        out["z_out_vs_capture_rel"] = float((zo - zr).norm() / zr.norm())
        print(f"z_out vs capture, real block: {out['z_out_vs_capture_rel']:.6e}", flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(out, indent=1) + "\n")
    print(f"layer_norm census: {census}", flush=True)
    print_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Where the fused SDPA's error LIVES, not just how big it is.

Every prior screen scored this kernel with `rel_rms` or PCC. Both are norms: they charge the same
price for every direction of the error vector. A fold does not. 1088 sequential triangle attentions
feed a residual+LayerNorm trunk, and the trunk's sensitivity to an error of fixed norm depends on
the error's STRUCTURE. So this splits the per-row error into

    E_par   the component along the fp64 reference row -- a per-row GAIN error, one scalar per row
    E_perp  what is left -- a DIRECTION error, per (row, channel)

and reports both, per arm, on real captured operands at several depths of a real fold.

Arms: the shipped materialised path, the fused kernel at the op default (what four of the six
models ship), the full {HiFi2,HiFi4} x {approx on,off} x {fp32_dest_acc on,off} cross so the three
bundled knobs are separated, and a torch-bf16 ceiling so "already at the floor" is a measured claim.

`--kchunk-probe` then varies k_chunk at a FIXED q_chunk on one captured call. That separates the two
sub-mechanisms that can make a per-channel error: a cross-chunk rescale that rounds the numerator
32x per row and the denominator once (grows with the boundary count) versus numerator and
denominator being reduced by different trees at all (flat in the boundary count).
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def err_struct(got, ref):
    """Split the per-row error into a gain component (along ref) and a direction component."""
    g_, r_ = got.double(), ref.double()
    e = g_ - r_
    num = (e * r_).sum(-1, keepdim=True)
    den = (r_ * r_).sum(-1, keepdim=True).clamp_min(1e-300)
    gain = num / den
    e_par = gain * r_
    e_perp = e - e_par
    nrm = r_.pow(2).sum().sqrt()
    return {
        "rel_total": float(e.pow(2).sum().sqrt() / nrm),
        "rel_par": float(e_par.pow(2).sum().sqrt() / nrm),
        "rel_perp": float(e_perp.pow(2).sum().sqrt() / nrm),
        "gain_mean": float(gain.mean()),
        "gain_sd": float(gain.std()),
    }


def fp64_ref(qh, kh, vh, bh, scale_inv, bias_scale_inv, chunk=4):
    """softmax(q@k^T * scale + bias * bias_scale) @ v in fp64 on the SAME bf16 operands."""
    q64, k64, v64 = qh.double(), kh.double(), vh.double()
    b64 = bh.double()
    out = torch.empty(qh.shape, dtype=torch.float64)
    for i in range(0, q64.shape[0], chunk):
        sc = q64[i:i + chunk] @ k64[i:i + chunk].transpose(-1, -2)
        bb = b64[i:i + chunk] if b64.shape[0] == q64.shape[0] else b64
        sc = sc * scale_inv + bb * bias_scale_inv
        out[i:i + chunk] = torch.softmax(sc, dim=-1) @ v64[i:i + chunk]
        del sc
    return out


def bf16_ceiling(qh, kh, vh, bh, scale_inv, bias_scale_inv, chunk=4):
    """The same maths in torch fp32 with a bf16-stored result: the floor both device arms share."""
    out = torch.empty(qh.shape, dtype=torch.bfloat16)
    for i in range(0, qh.shape[0], chunk):
        sc = qh[i:i + chunk].float() @ kh[i:i + chunk].float().transpose(-1, -2)
        bb = bh[i:i + chunk] if bh.shape[0] == qh.shape[0] else bh
        sc = sc * scale_inv + bb.float() * bias_scale_inv
        out[i:i + chunk] = (torch.softmax(sc, dim=-1) @ vh[i:i + chunk].float()).bfloat16()
        del sc
    return out


class Reservoir:
    """Keep `n` captures evenly spread over an unknown-length call stream.

    Take every `stride`-th call; when the list reaches 2n, drop every other entry and double the
    stride. The survivors are always indices 0, s, 2s, ... so the spread is uniform over whatever
    the total turns out to be, and no pre-count fold is needed.
    """

    def __init__(self, n):
        self.n, self.stride, self.items = n, 1, []

    def want(self, i):
        return i % self.stride == 0

    def add(self, rec):
        self.items.append(rec)
        if len(self.items) >= 2 * self.n:
            del self.items[1::2]
            self.stride *= 2


def capture_rf3(args, T, ttnn):
    """Spy on the materialised path inside a real RF3 recycler call and keep several depths."""
    from tt_bio.rf3.featurize import featurize
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from perf.rf3.tt_rf3_bench import net_config

    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]
    f = fo["feats"]
    cfg = net_config(args.ckpt)
    device = T.get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=50, with_confidence=False)

    res = Reservoir(args.n_keep)
    orig = T._fp32_softmax_attention
    counter = [0]

    def spy(q, k, v, bias, scale_inv, compute_kernel_config, out_dtype=ttnn.bfloat16,
            bias_scale_inv=None, accurate_softmax=False):
        shp = tuple(int(d) for d in q.shape)
        # Triangle attention is the only caller whose batch dim IS the sequence dim.
        if len(shp) == 4 and shp[0] == shp[2] and shp[0] > 1:
            i = counter[0]
            counter[0] += 1
            if res.want(i):
                bsi = bias_scale_inv if bias_scale_inv else scale_inv
                matched = abs(scale_inv - bsi) < 1e-12
                qt, kt, vt = ttnn.to_torch(q), ttnn.to_torch(k), ttnn.to_torch(v)
                bt = ttnn.to_torch(bias)
                sub = torch.arange(0, qt.shape[0], max(1, qt.shape[0] // args.rows))
                res.add(dict(
                    call=i, site="pairformer" if matched else "msa_or_template",
                    q=qt[sub].clone(), k=kt[sub].clone(), v=vt[sub].clone(),
                    bias=(bt[sub].clone() if bt.shape[0] == qt.shape[0] else bt.clone()),
                    scale_inv=float(scale_inv), bias_scale_inv=float(bsi),
                    shape=list(shp)))
                del qt, kt, vt, bt
        return orig(q, k, v, bias, scale_inv, compute_kernel_config, out_dtype, bias_scale_inv,
                    accurate_softmax)

    T._fp32_softmax_attention = spy
    try:
        host = HostInputs.build(f, device)
        s_inputs, s_init, z_init = tt.feature_initializer(
            host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
            host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
            host.token_feats, host.relpos_feat, host.bond_feat)
        tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        tt.recycler(host, tmpl, host.msa_stack[0], s_inputs, s_init, z_init, s, z)
    finally:
        T._fp32_softmax_attention = orig
    return res.items, counter[0], device, kcfg


def capture_protenix(args, T, ttnn):
    """Spy on the FUSED path inside a real Protenix-v2 trunk -- the op-default models' own inputs."""
    res = Reservoir(args.n_keep)
    counter = [0]
    orig = T._tri_att_sdpa

    def spy(q, k, v, bias, scale):
        shp = tuple(int(d) for d in q.shape)
        if len(shp) == 4 and shp[0] == shp[2] and shp[0] > 1:
            i = counter[0]
            counter[0] += 1
            if res.want(i):
                qt, kt, vt = ttnn.to_torch(q), ttnn.to_torch(k), ttnn.to_torch(v)
                bt = ttnn.to_torch(bias)
                sub = torch.arange(0, qt.shape[0], max(1, qt.shape[0] // args.rows))
                res.add(dict(
                    call=i, site="pairformer",
                    q=qt[sub].clone(), k=kt[sub].clone(), v=vt[sub].clone(),
                    bias=(bt[sub].clone() if bt.shape[0] == qt.shape[0] else bt.clone()),
                    scale_inv=float(scale), bias_scale_inv=float(scale),
                    shape=list(shp)))
                del qt, kt, vt, bt
        return orig(q, k, v, bias, scale)

    T._tri_att_sdpa = spy
    try:
        from tt_bio import protenix as px
        device = T.get_device()
        px.predict(args.px_input, out_dir=args.px_work, device=device,
                   seed=args.seed, n_sample=1, n_step=2, n_cycle=1)
    except Exception as exc:  # noqa: BLE001
        print(f"protenix capture raised after {counter[0]} calls: "
              f"{type(exc).__name__}: {exc}", flush=True)
    finally:
        T._tri_att_sdpa = orig
    kcfg = ttnn.init_device_compute_kernel_config(
        T.get_device().arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    return res.items, counter[0], T.get_device(), kcfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("rf3", "protenix"), default="rf3")
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--px-input", default=str(REPO / "perf/size512/fixtures/cdk2x2_298.yaml"))
    ap.add_argument("--px-work", default="/home/ttuser/errstruct_px")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", type=int, default=64,
                    help="batch rows kept per capture; each row is an independent attention, so a "
                         "stride subsample is an unbiased estimator of the per-row error structure "
                         "and it costs 8x less fp64 host time at 512 aa")
    ap.add_argument("--n-keep", type=int, default=8,
                    help="captures kept, spread evenly over the call stream by a decimating "
                         "reservoir -- no pre-count fold needed")
    ap.add_argument("--kchunk-probe", action="store_true",
                    help="vary k_chunk at a FIXED q_chunk: rel_perp growing with the\nboundary count says the cross-chunk rescale carries the direction error, flat says\nthe numerator/denominator reduction-tree mismatch does")
    ap.add_argument("--probe-calls", default="",
                    help="comma-separated captured call indices to k-chunk probe; default the first")
    ap.add_argument("--capture-out", default=None, help="torch.save the capture here for reuse")
    ap.add_argument("--capture-in", default=None, help="skip the fold, load a capture")
    ap.add_argument("--lens", default="",
                    help="comma-separated sequence lengths to re-score the first capture at, handed "
                         "over RAW: a length that is not a multiple of 32 keeps its ragged tail, "
                         "which is exactly what the fold does and what every prior screen padded "
                         "away")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().is_relative_to(REPO), \
        f"wrong tree: {T.__file__} (need PYTHONPATH={REPO})"
    import tt_bio.triatt_sdpa as _ts

    if args.capture_in:
        blob = torch.load(args.capture_in, weights_only=False)
        grabs, seen = blob["grabs"], blob["seen"]
        device = T.get_device()
        kcfg = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
    else:
        cap = capture_rf3 if args.model == "rf3" else capture_protenix
        grabs, seen, device, kcfg = cap(args, T, ttnn)
        if args.capture_out:
            torch.save({"grabs": grabs, "seen": seen, "model": args.model, "aa": args.aa},
                       args.capture_out)
            print(f"wrote capture {args.capture_out} ({len(grabs)} calls of {seen})", flush=True)
    assert grabs, "no triangle-attention call captured"
    print(f"captured {len(grabs)} of {seen} calls: "
          f"{[(g['call'], g['site'], g['shape'][2]) for g in grabs]}", flush=True)

    HF = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    _probe_calls = {int(x) for x in args.probe_calls.split(",") if x.strip()}
    report = {"model": args.model, "aa": args.aa, "rows_kept": args.rows,
              "calls_seen": seen, "calls": [], "kchunk_probe": []}

    for g in grabs:
        qh, kh, vh, bh = g["q"], g["k"], g["v"], g["bias"]
        si, bsi = g["scale_inv"], g["bias_scale_inv"]
        ref = fp64_ref(qh, kh, vh, bh, si, bsi)
        ceil = bf16_ceiling(qh, kh, vh, bh, si, bsi)
        qd, kd, vd, bd = up(qh), up(kh), up(vh), up(bh)
        # The fused kernel adds the bias BEFORE applying scale, so it wants the bias pre-baked by
        # sqrt(head_dim). Where the model already did that the two scales agree and this is a no-op.
        bias_mul = bsi / si
        bd_f = bd if abs(bias_mul - 1.0) < 1e-12 else ttnn.multiply(bd, bias_mul)

        rows = {"bf16_ceiling": err_struct(ceil.double(), ref)}

        def score(name, fn):
            try:
                o = fn()
            except Exception as exc:  # noqa: BLE001
                rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
                return
            if o is None:
                rows[name] = {"error": "declined"}
                return
            got = ttnn.to_torch(o).double()
            ttnn.deallocate(o)
            rows[name] = err_struct(got, ref)
            del got

        score("materialised", lambda: T._fp32_softmax_attention(
            qd, kd, vd, bd, scale_inv=si, compute_kernel_config=kcfg,
            out_dtype=ttnn.bfloat16, bias_scale_inv=bsi))
        score("fused_default", lambda: T._tri_att_sdpa(qd, kd, vd, bd_f, si))
        for fid in ("HiFi2", "HiFi4"):
            for approx in (True, False):
                for acc in (False, True):
                    ckc = (HF[fid], approx, acc, False)

                    def run(ckc=ckc):
                        T._TRIATT_FUSED_HIFI_CKC = ckc
                        T._TRIATT_HIFI_OVER_L1.clear()
                        return T._tri_att_sdpa_hifi(qd, kd, vd, bd_f, si)
                    score(f"{fid}_ap{int(approx)}_acc{int(acc)}", run)

        for nm, r in sorted(rows.items()):
            if "error" in r:
                print(f"  call {g['call']:5d} {nm:24s} {r['error']}", flush=True)
            else:
                print(f"  call {g['call']:5d} {nm:24s} total {r['rel_total']:.5e}  "
                      f"par {r['rel_par']:.5e}  perp {r['rel_perp']:.5e}  "
                      f"gain {r['gain_mean']:+.3e}+-{r['gain_sd']:.3e}", flush=True)
        report["calls"].append({"call": g["call"], "site": g["site"], "shape": g["shape"],
                                "scale_inv": si, "bias_scale_inv": bsi, "arms": rows})

        if args.kchunk_probe and (g is grabs[0] if not args.probe_calls
                                  else g["call"] in _probe_calls):
            S = int(qd.shape[2])
            padded = T._padded_sdpa_len(S)
            # q_chunk PINNED to the shipped pick. Widening q against a wide k is the one pair the
            # production ladder retires over L1, and a probe that dies there measures nothing.
            q_chunk = T._sdpa_chunks_shipped(S, S)[0]
            cands = [c for c in (padded, padded // 2, padded // 4, padded // 8, padded // 16, 64)
                     if c >= 32 and padded % c == 0]
            for kc in sorted(set(cands), reverse=True):
                T._TRIATT_HIFI_OVER_L1.clear()
                served, o = "fused", None
                try:
                    o = _ts.sdpa(qd, kd, vd, bd_f, si, q_chunk, kc,
                                 ckc_default=(HF["HiFi4"], False, True, False))
                except Exception as exc:  # noqa: BLE001
                    print(f"  KPROBE k={kc:5d} fused refused: {str(exc)[:80]}", flush=True)
                if o is None:
                    served = "stock"
                    try:
                        o = ttnn.transformer.scaled_dot_product_attention(
                            qd, kd, vd, attn_mask=bd_f, is_causal=False, scale=si,
                            program_config=T._sdpa_program_config(q_chunk, kc))
                    except Exception as exc:  # noqa: BLE001
                        print(f"  KPROBE k={kc:5d} SKIPPED: {str(exc)[:80]}", flush=True)
                        continue
                st = err_struct(ttnn.to_torch(o).double(), ref)
                ttnn.deallocate(o)
                st.update(call=g["call"], k_chunk=kc, q_chunk=q_chunk,
                          boundaries=padded // kc - 1, served=served)
                report["kchunk_probe"].append(st)
                print(f"  KPROBE call {g['call']:5d} k={kc:5d} bnd {padded // kc - 1:3d} "
                      f"{served:6s} total {st['rel_total']:.5e}  par {st['rel_par']:.5e}  "
                      f"perp {st['rel_perp']:.5e}", flush=True)

        for t in (qd, kd, vd, bd):
            ttnn.deallocate(t)
        del ref, ceil

    if args.lens:
        g = grabs[0]
        qh, kh, vh, bh = g["q"], g["k"], g["v"], g["bias"]
        si, bsi = g["scale_inv"], g["bias_scale_inv"]
        report["len_ladder"] = []
        for n in [int(x) for x in args.lens.split(",") if x.strip()]:
            if n > qh.shape[2]:
                continue
            # Each batch row of a triangle attention is an independent attention, so taking the
            # first min(n, rows) of them and cutting the key axis to n is a valid call at length n.
            nb = min(n, qh.shape[0])
            qn, kn, vn = qh[:nb, :, :n], kh[:nb, :, :n], vh[:nb, :, :n]
            bn = bh[:nb, :, :n, :n] if bh.shape[0] == qh.shape[0] else bh[:, :, :n, :n]
            refn = fp64_ref(qn, kn, vn, bn, si, bsi)
            ceiln = bf16_ceiling(qn, kn, vn, bn, si, bsi)
            qdn, kdn, vdn, bdn = up(qn), up(kn), up(vn), up(bn)
            bias_mul = bsi / si
            bdn_f = bdn if abs(bias_mul - 1.0) < 1e-12 else ttnn.multiply(bdn, bias_mul)
            row = {"n": n, "tile_aligned": n % 32 == 0, "batch": nb,
                   "bf16_ceiling": err_struct(ceiln.double(), refn)}
            for nm, fn in (
                ("materialised", lambda: T._fp32_softmax_attention(
                    qdn, kdn, vdn, bdn, scale_inv=si, compute_kernel_config=kcfg,
                    out_dtype=ttnn.bfloat16, bias_scale_inv=bsi)),
                ("fused_default", lambda: T._tri_att_sdpa(qdn, kdn, vdn, bdn_f, si)),
                ("fused_hifi_acc", lambda: (
                    setattr(_ts, "_CKC_OVERRIDE", (HF["HiFi4"], False, True, False)),
                    T._tri_att_sdpa(qdn, kdn, vdn, bdn_f, si))[1]),
            ):
                try:
                    o = fn()
                except Exception as exc:  # noqa: BLE001
                    row[nm] = {"error": f"{type(exc).__name__}: {str(exc)[:90]}"}
                    continue
                finally:
                    _ts._CKC_OVERRIDE = None
                if o is None:
                    row[nm] = {"error": "declined"}
                    continue
                got = ttnn.to_torch(o).double()
                ttnn.deallocate(o)
                row[nm] = err_struct(got, refn)
                del got
            for a, b in (("fused_default", "materialised"), ("fused_hifi_acc", "materialised")):
                if "rel_total" in row.get(a, {}) and "rel_total" in row.get(b, {}):
                    row[f"{a}_over_mat_total"] = row[a]["rel_total"] / row[b]["rel_total"]
                    row[f"{a}_over_mat_perp"] = row[a]["rel_perp"] / row[b]["rel_perp"]
            report["len_ladder"].append(row)
            al = "aligned" if row["tile_aligned"] else "RAGGED "
            def _t(k):
                return row[k].get("rel_total", float("nan")) if isinstance(row.get(k), dict) else float("nan")
            print(f"  LEN n={n:5d} {al} mat {_t('materialised'):.4e}  "
                  f"fused_def {_t('fused_default'):.4e} (x{row.get('fused_default_over_mat_total', float('nan')):.3f})  "
                  f"fused_hifi {_t('fused_hifi_acc'):.4e} (x{row.get('fused_hifi_acc_over_mat_total', float('nan')):.3f})  "
                  f"ceil {row['bf16_ceiling']['rel_total']:.4e}", flush=True)
            for t in (qdn, kdn, vdn, bdn):
                ttnn.deallocate(t)
            del refn, ceiln

    # P1: does the fused arm carry MORE direction error while carrying LESS total error?
    p1 = []
    for c in report["calls"]:
        a, b = c["arms"].get("materialised", {}), c["arms"].get("fused_default", {})
        if "rel_perp" in a and "rel_perp" in b:
            p1.append({"call": c["call"], "site": c["site"],
                       "perp_ratio": b["rel_perp"] / a["rel_perp"],
                       "total_ratio": b["rel_total"] / a["rel_total"]})
    report["p1"] = p1
    hits = sum(1 for r in p1 if r["perp_ratio"] > 1.0 and r["total_ratio"] < 1.0)
    report["p1_verdict"] = f"{hits}/{len(p1)} calls have perp_ratio>1 and total_ratio<1"
    print("P1: " + report["p1_verdict"], flush=True)
    for r in p1:
        print(f"   call {r['call']:5d} {r['site']:16s} perp x{r['perp_ratio']:.3f}  "
              f"total x{r['total_ratio']:.3f}", flush=True)
    report["stats"] = {
        "fused_hifi": dict(T.TRIATT_FUSED_HIFI_STATS),
        "triatt_served_declined": list(_ts.STATS),
        "rejects": {str(k): v for k, v in _ts.REJECTS.items()},
        "chunk_picks": {f"{a}x{b}": v for (a, b), v in T.SDPA_CHUNK_PICKS.items()},
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("wrote " + args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

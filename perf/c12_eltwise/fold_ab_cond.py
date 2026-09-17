#!/usr/bin/env python3
"""Fold-level A/B for TT_BIO_FUSE_COND_MULADD, interleaved rep by rep in one session.

Arms, in this order every rep, so a co-tenant's load cancels in the pairing:

    off   main's shipped chain: multiply_(a, s_scale, SIGMOID) ; add_(., s_bias)
    aa    `off` again -- this session's own A/A floor
    on    the gate's sigmoid in the s_scale matmul epilogue, then one addcmul

`aa/off` is the noise floor for THIS session. A ratio smaller than its own floor is not a
result and the session is discarded rather than explained. The clock is sampled DURING the
run from the host telemetry log, and the host's load average is recorded with every number,
because the release gate shares this box.

Accuracy is scored paired: the last `off` and the last `on` CIF, same seed, same fixture,
Kabsch-superposed all-atom RMSD. The fused arm is not bit-exact by construction (the chain
packs the product to bf16 and reloads it, the fused op keeps it in an fp32 SFPU register),
so the number that matters is the deviation against the 512 aa kill bar with the seed floor
beside it, not a digest match.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

TEL = "/home/ttuser/qbcard/cardtel.tsv"


def clock_during(t0, t1, card):
    try:
        head, vals = None, []
        with open(TEL) as f:
            for line in f:
                if line.startswith("#epoch"):
                    head = line.rstrip("\n").split("\t")
                    continue
                if line.startswith("#") or head is None:
                    continue
                p = line.rstrip("\n").split("\t")
                if not (t0 <= float(p[0]) <= t1):
                    continue
                i = head.index("c%d_tt_aiclk" % card)
                if i < len(p) and p[i].strip():
                    vals.append(int(p[i]))
        if not vals:
            return {"samples": 0}
        return {"samples": len(vals), "min_MHz": min(vals), "max_MHz": max(vals),
                "median_MHz": st.median(vals),
                "pct_at_1350": round(100.0 * sum(v >= 1350 for v in vals) / len(vals), 2)}
    except Exception as e:
        return {"error": str(e)}


def coords(cif_path):
    import torch
    xs = []
    for line in Path(cif_path).read_text().splitlines():
        p = line.split()
        if len(p) > 12 and p[0] in ("ATOM", "HETATM"):
            try:
                xs.append([float(p[10]), float(p[11]), float(p[12])])
            except ValueError:
                continue
    return torch.tensor(xs, dtype=torch.float64)


def kabsch_rmsd(P, Q):
    import torch
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    H = P.T @ Q
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=P.dtype))
    Pr = (Vt.T @ D @ U.T @ P.T).T
    return float(torch.sqrt(((Pr - Q) ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--size", default="512")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "2")))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "perf/c12_eltwise/runs/fold_ab_cond_512.json")
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

    def fold(fused):
        EF.FUSE_COND_MULADD = fused
        total, _metrics = one_fold()
        cifs = sorted(struct_dir.glob("*.cif"))
        blob = b"".join(p.read_bytes() for p in cifs)
        return total, hashlib.sha256(blob).hexdigest()[:16], cifs

    t_start = time.time()
    # One cold fold PER ARM. The first call is the full predict_one, and each arm compiles its
    # own programs (linear with and without the sigmoid epilogue, addcmul vs multiply+add), so
    # warming only one arm would charge the other's compile to its first timed rep.
    for _n, _f in (("off", False), ("on", True)):
        EF.FUSE_COND_MULADD = _f
        _t, _m = one_fold()
        print("cold %-4s %8.3f s" % (_n, _t), flush=True)

    arms = [("off", False), ("aa", False), ("on", True)]
    acc = {n: [] for n, _ in arms}
    digest = {}
    saved = {}
    for rep in range(a.reps):
        for name, fused in arms:
            dt, dg, cifs = fold(fused)
            acc[name].append(dt)
            digest.setdefault(name, set()).add(dg)
            if rep == a.reps - 1 and name in ("off", "on"):
                dst = a.out.parent / ("cif_%s" % name)
                dst.mkdir(parents=True, exist_ok=True)
                for c in cifs:
                    (dst / c.name).write_bytes(c.read_bytes())
                saved[name] = sorted(str(dst / c.name) for c in cifs)
            print("rep %d %-4s %8.3f s  digest %s" % (rep, name, dt, dg), flush=True)
    t_end = time.time()
    EF.FUSE_COND_MULADD = False

    med = {n: st.median(v) for n, v in acc.items()}
    rmsd = None
    if "off" in saved and "on" in saved and len(saved["off"]) == len(saved["on"]):
        try:
            rs = [kabsch_rmsd(coords(o), coords(n))
                  for o, n in zip(saved["off"], saved["on"])]
            rmsd = {"per_model_A": rs, "max_A": max(rs), "median_A": st.median(rs)}
        except Exception as e:
            rmsd = {"error": str(e)}

    out = {
        "model": a.model, "size": a.size, "card": a.card, "reps": a.reps,
        "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
        "clock_during": clock_during(t_start, t_end, a.card),
        "loadavg_end": os.getloadavg(), "cotenant": "v0.9.0 release gate on cards 0/1",
        "seconds": acc, "median_s": med,
        "aa_floor": med["aa"] / med["off"],
        "ratio_on_over_off": med["on"] / med["off"],
        "delta_s": med["off"] - med["on"],
        "digests": {k: sorted(v) for k, v in digest.items()},
        "bitexact_on_vs_off": sorted(digest.get("on", set())) == sorted(digest.get("off", set())),
        "rmsd_on_vs_off": rmsd,
        "kill_bar_A": 0.60, "seed_scatter_floor_A": 1.84,
        "meta": {k: meta[k] for k in ("hardware", "grid", "recycling_steps", "diffusion_samples")
                 if k in meta},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))

    print("\nclock during: %s" % json.dumps(out["clock_during"]))
    print("loadavg %s, co-tenant: %s" % (out["loadavg_end"], out["cotenant"]))
    print("off %.3f s   aa %.3f s   on %.3f s" % (med["off"], med["aa"], med["on"]))
    print("A/A floor %.4f   on/off %.4f   delta %+.3f s"
          % (out["aa_floor"], out["ratio_on_over_off"], out["delta_s"]))
    print("bit-exact %s   RMSD %s" % (out["bitexact_on_vs_off"], json.dumps(rmsd)))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()

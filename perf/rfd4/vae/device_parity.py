"""Device parity of the RFD4 autoencoder against upstream's float64 goldens (gen_goldens.py).

    TT_VISIBLE_DEVICES=<chip> timeout 1800 python perf/rfd4/vae/device_parity.py \
        --goldens <dir> --ckpt <rfd4_proteina_ae-ema.ckpt> --out <json>

Grades, per case:
  encoder   mean / log_scale / z_latent: max_rel_err (max |d| over emitted tokens / max |ref|), rel_l2
  decoder   seq_logits on legal classes: max_rel_err; residue argmax agreement, with the reference
            top-2 margin of every disagreement (a disagreement on a near-tie is not an error);
            softmax distribution: mean / max total variation and mean KL(ref || tt)
            coordinates: RMSD and max deviation in Angstrom over the reference atom mask
            atom presence: max_rel_err and sign agreement (it thresholds at 0)
  roundtrip TT encode -> TT decode against upstream encode -> decode on the same noise draw
Timing: warm encode and decode seconds, with the AICLK sampled during the timed calls.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

BAR = {
    "max_rel_err": 2e-2,          # continuous outputs: latents, logits, presence logits
    "coord_rmsd_A": 0.05,         # decoded all-atom coordinates vs float64, Angstrom
    "argmax_agree": 0.99,         # residue type agreement over emitted tokens
    "tv_mean": 1e-2,              # mean total variation of the 147-class distribution
    # Real captured structures: the TT reconstruction must be as good as upstream's, not merely close
    # to it in tensor norm. Upstream's own all-atom RMSD is the yardstick, so these are DELTAS.
    "d_rmsd_A": 0.02,
    "d_seq_recovery": -0.01,
}


def rel(a, b, mask=None):
    a, b = a.double(), b.double()
    if mask is not None:
        a, b = a[mask], b[mask]
    d = (a - b)
    return {"max_rel_err": float(d.abs().max() / b.abs().max().clamp_min(1e-30)),
            "rel_l2": float(d.norm() / b.norm().clamp_min(1e-30))}


def grade_decoder(tt: dict, ref: dict, ref_fin: dict, tt_fin: dict, legal: torch.Tensor) -> dict:
    emit = ref["emit_mask"]
    lg_t, lg_r = tt["seq_logits"].double(), ref["seq_logits"].double()
    out = {"seq_logits": rel(lg_t[legal], lg_r[legal])}
    neg = torch.finfo(torch.float64).min
    p_r = torch.softmax(lg_r.masked_fill(~legal, neg), -1)[emit]
    p_t = torch.softmax(lg_t.masked_fill(~legal, neg), -1)[emit]
    tv = 0.5 * (p_r - p_t).abs().sum(-1)
    kl = (p_r * (p_r.clamp_min(1e-300).log() - p_t.clamp_min(1e-300).log())).sum(-1)
    am_t, am_r = tt["aatype_max"][emit], ref["aatype_max"][emit]
    dis = (am_t != am_r).nonzero().flatten()
    top2 = p_r.topk(2, -1).values
    out["residue"] = {
        "argmax_agree": float((am_t == am_r).double().mean()), "n_tokens": int(emit.sum()),
        "disagree_ref_margin": [float(top2[i, 0] - top2[i, 1]) for i in dis.tolist()],
        "tv_mean": float(tv.mean()), "tv_max": float(tv.max()), "kl_mean": float(kl.mean()),
        "final_residue_type_agree": float((tt_fin["residue_type"] == ref_fin["residue_type"])[emit].double().mean()),
    }
    am = ref_fin["atom_mask"]
    d = (tt["coors_nm"].double() - ref["coors_nm"].double())[am] * 10.0
    dist = d.norm(dim=-1)
    out["coords"] = {"rmsd_A": float(dist.pow(2).mean().sqrt()), "max_dev_A": float(dist.max()), "n_atoms": int(am.sum())}
    pt, pr = tt["atom_presence_logits"].double()[emit], ref["atom_presence_logits"].double()[emit]
    out["presence"] = dict(rel(pt, pr), sign_agree=float(((pt > 0) == (pr > 0)).double().mean()))
    out["atom_mask_agree"] = float((tt_fin["atom_mask"] == ref_fin["atom_mask"]).double().mean())
    return out


def recon(fin: dict, batch: dict) -> dict | None:
    """Reconstruction against the INPUT structure: all-atom RMSD (no alignment, as upstream's
    `rmsd_no_align_a37_ang`) and sequence recovery. Only a real captured batch carries both."""
    if "residue_type" not in batch or "coords_nm" not in batch:
        return None
    m = fin["atom_mask"] & batch["atom37_mask"].bool()
    d = (fin["coors_nm"].double() - batch["coords_nm"].double())[m] * 10.0
    emit = fin["emit_mask"]
    return {"all_atom_rmsd_A": float(d.norm(dim=-1).pow(2).mean().sqrt()), "n_atoms": int(m.sum()),
            "seq_recovery": float((fin["residue_type"] == batch["residue_type"])[emit].double().mean()),
            "n_tokens": int(emit.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from perf.clocksample import during
    from tt_bio.rfd4 import ae_fold as AF
    from tt_bio.rfd4.autoencoder import RFD4AutoEncoder

    t0 = time.time()
    ae = RFD4AutoEncoder.from_checkpoint(a.ckpt)
    rec = {"bar": BAR, "load_s": round(time.time() - t0, 2), "cases": {}}
    tab = ae.tab
    for path in sorted(Path(a.goldens).glob("*.pt")):
        g = torch.load(path, map_location="cpu", weights_only=False)
        batch, noise = g["batch"], g["noise"]
        legal = tab["allowed"][batch["token_type"].long()] & g["decoder"]["emit_mask"][..., None]
        c = {"n_tokens": int(batch["padding_mask"].shape[1]), "batch": int(batch["padding_mask"].shape[0])}
        e = ae.encode(batch, noise)
        em = g["encoder"]["emit_mask"]
        c["encoder"] = {k: rel(e[k], g["encoder"][k], em) for k in ("mean", "log_scale", "z_latent")}
        d = ae.decode_raw(batch)
        fin = AF.finish_decode(d, batch, tab)
        c["decoder"] = grade_decoder(d, g["decoder"], g["decode"], fin, legal)
        rb = dict(batch, z_latent=e["z_latent"])
        d2 = ae.decode_raw(rb)
        fin2 = AF.finish_decode(d2, rb, tab)
        c["roundtrip"] = grade_decoder(d2, g["decoder"], g["decode"], fin2, legal)
        # Product-level: does the TT autoencoder reconstruct the input as well as upstream does?
        r_ref, r_tt, r_rt = recon(g["decode"], batch), recon(fin, batch), recon(fin2, batch)
        if r_ref is not None:
            c["recon"] = {"upstream": r_ref, "tt_decode": r_tt, "tt_roundtrip": r_rt,
                          "d_rmsd_A": r_tt["all_atom_rmsd_A"] - r_ref["all_atom_rmsd_A"],
                          "d_seq_recovery": r_tt["seq_recovery"] - r_ref["seq_recovery"]}
        with during(period=1.0) as clk:
            ts = {"encode": [], "decode": []}
            for _ in range(a.reps):
                t = time.perf_counter(); ae.encode(batch, noise); ts["encode"].append(time.perf_counter() - t)
                t = time.perf_counter(); ae.decode_raw(batch); ts["decode"].append(time.perf_counter() - t)
        c["timing_s"] = {k: {"min": min(v), "median": sorted(v)[len(v) // 2]} for k, v in ts.items()}
        c["aiclk"] = clk.summary()
        rec["cases"][g["case"]] = c
        print(g["case"], json.dumps(c)[:600], flush=True)

    def worst(key_path):
        vals = []
        for c in rec["cases"].values():
            v = c
            for k in key_path:
                v = v[k]
            vals.append(v)
        return vals

    cont = [x for p in (("encoder", "mean"), ("encoder", "log_scale"), ("encoder", "z_latent"),
                        ("decoder", "seq_logits"), ("decoder", "presence"))
            for x in worst(p + ("max_rel_err",))]
    rec["max_rel_err"] = max(cont)
    rec["coord_rmsd_A"] = max(worst(("decoder", "coords", "rmsd_A")))
    rec["argmax_agree"] = min(worst(("decoder", "residue", "argmax_agree")))
    rec["tv_mean"] = max(worst(("decoder", "residue", "tv_mean")))
    real = [c["recon"] for c in rec["cases"].values() if "recon" in c]
    ok = (rec["max_rel_err"] <= BAR["max_rel_err"] and rec["coord_rmsd_A"] <= BAR["coord_rmsd_A"]
          and rec["argmax_agree"] >= BAR["argmax_agree"] and rec["tv_mean"] <= BAR["tv_mean"])
    if real:
        rec["d_rmsd_A"] = max(r["d_rmsd_A"] for r in real)
        rec["d_seq_recovery"] = min(r["d_seq_recovery"] for r in real)
        rec["n_real_cases"] = len(real)
        ok = ok and rec["d_rmsd_A"] <= BAR["d_rmsd_A"] and rec["d_seq_recovery"] >= BAR["d_seq_recovery"]
    rec["verdict"] = "PASS" if ok else "FAIL"
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print("VERDICT", rec["verdict"], json.dumps({k: rec.get(k) for k in
          ("max_rel_err", "coord_rmsd_A", "argmax_agree", "tv_mean", "d_rmsd_A", "d_seq_recovery")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

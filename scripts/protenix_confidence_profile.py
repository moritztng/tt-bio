"""Re-profile the Protenix-v2 confidence head on THIS host (real predict path).

The qb1 BH measurement (memory protenix-accel-ceiling, 2026-07-08) found the
confidence head host-bound at the large-N end (N=256: 486 ms total, device
Pairformer 116 ms vs host up/download + z-embed + heads 370 ms) and projected a
device-resident z-embed + heads port at ~4% e2e (n_sample=25). qb1 is off-limits
this task and its percentages need not transfer to a different card, so this
harness re-derives the split fresh on the local card with the one cached real
target (the gold NT=38 fold), then characterizes how the split scales with N by
padding the real target's tensors.

Reports (warm, device-synchronized):
  - e2e fold wall-clock (trunk, diffusion, confidence) at the cached target
  - confidence internal split: z-embed (host), upload z (N,N,256), device
    Pairformer, download (s_single + zf), heads (host), total
  - the same split at padded N (128, 256) so the device-port lever can be
    sized at the N where qb1 saw it

No reference CLI is run; the cached feats pkls + the production ckpt are reused.
"""
import os, sys, time
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle, torch, torch.nn.functional as F, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN as CORE
from tt_bio.protenix import Protenix, ConfidenceHead

CKPT = os.environ.get("PROTENIX_CKPT", "/home/moritz/.boltz/protenix-v2.pt")
IFE = os.environ.get("PROTENIX_IFE", "/home/moritz/protenix_ife_gold.pkl")
TG = os.environ.get("PROTENIX_TG", "/home/moritz/protenix_trunkin_gold.pkl")
REF = os.environ.get("PROTENIX_REF", "/home/moritz/protenix_ref_out.pkl")


def _load_feats():
    ife = pickle.load(open(IFE, "rb"))
    F = ife["feat"]
    d = pickle.load(open(REF, "rb"))
    tfeat = d["intermediates"]["template_embedder"]["in"][0]
    tg = pickle.load(open(TG, "rb"))
    rfeat = d.get("feat", {})
    feats = {
        "ref_pos": F["ref_pos"], "ref_charge": F["ref_charge"], "ref_mask": F["ref_mask"],
        "ref_element": F["ref_element"], "ref_atom_name_chars": F["ref_atom_name_chars"],
        "d_lm": F["d_lm"], "v_lm": F["v_lm"], "atom_to_token_idx": F["atom_to_token_idx"],
        "restype": F["restype"], "profile": F["profile"], "deletion_mean": F["deletion_mean"],
        "mask_trunked": ife["mask_trunked"],
        "relp": tg["relp"], "token_bonds": tg["token_bonds"],
        "template_aatype": tfeat["template_aatype"], "template_distogram": tfeat["template_distogram"],
        "template_pseudo_beta_mask": tfeat["template_pseudo_beta_mask"],
        "template_unit_vector": tfeat["template_unit_vector"],
        "template_backbone_frame_mask": tfeat["template_backbone_frame_mask"],
        "msa": tfeat["msa"], "has_deletion": tfeat["has_deletion"],
        "deletion_value": tfeat["deletion_value"], "asym_id": tfeat["asym_id"],
        # confidence-head keys (from the reference feat dict)
        "distogram_rep_atom_mask": rfeat["distogram_rep_atom_mask"],
        "atom_to_tokatom_idx": rfeat["atom_to_tokatom_idx"],
        "ref_space_uid": rfeat.get("ref_space_uid", torch.zeros(F["ref_pos"].shape[0], dtype=torch.long)),
    }
    gold = d.get("pred", {})
    return feats, F, gold


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm() + 1e-12))


def _sync(dev):
    ttnn.synchronize_device(dev)


def _time(fn, dev, warm=1, reps=3):
    for _ in range(warm):
        fn()
    _sync(dev)
    ts = time.time()
    for _ in range(reps):
        fn()
    _sync(dev)
    return (time.time() - ts) / reps


def confidence_split(ch, s_inputs, s_trunk, z_trunk, coords, feats, dev):
    """Run ConfidenceHead.confidence with per-component timing (warm). Returns
    (split_dict_ms, conf_dict). Mirrors ConfidenceHead.confidence exactly."""
    w = ch._w
    g = ch._g; bias = ch._bias
    N = s_trunk.shape[0]

    def zembed():
        s_t = F.layer_norm(torch.clamp(s_trunk, -512, 512), (384,)) * g("input_strunk_ln.weight") + bias("input_strunk_ln.bias")
        z = (z_trunk + F.linear(s_inputs, g("linear_no_bias_s1.weight")).unsqueeze(1)
             + F.linear(s_inputs, g("linear_no_bias_s2.weight")).unsqueeze(0))
        mask = feats["distogram_rep_atom_mask"].bool()
        xr = coords.reshape(-1, 3)[mask]
        d = torch.cdist(xr, xr)
        oh = ((d.unsqueeze(-1) >= g("lower_bins")) & (d.unsqueeze(-1) < g("upper_bins"))).float()
        z = z + F.linear(oh, g("linear_no_bias_d.weight")) + F.linear(d.unsqueeze(-1), g("linear_no_bias_d_wo_onehot.weight"))
        return s_t, z

    _sync(dev)
    t = {}
    t["zembed_host"] = _time(zembed, dev, warm=0, reps=5) * 1000
    s_t, z = zembed()

    T = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    t["upload_z"] = _time(lambda: T(z.unsqueeze(0)), dev, warm=0, reps=5) * 1000
    t["upload_s"] = _time(lambda: T(s_t.unsqueeze(0)), dev, warm=0, reps=5) * 1000
    zt, st = T(z.unsqueeze(0)), T(s_t.unsqueeze(0))

    def pf():
        so, zo = ch.pf(st, zt)
        return so, zo
    t["pairformer_dev"] = _time(pf, dev, warm=1, reps=5) * 1000
    so, zo = pf()

    def dl():
        s_single = torch.Tensor(ttnn.to_torch(so)).float().reshape(N, 384)
        zf = torch.Tensor(ttnn.to_torch(zo)).float().reshape(N, N, -1)
        return s_single, zf
    t["download_dev"] = _time(dl, dev, warm=0, reps=5) * 1000
    s_single, zf = dl()

    def heads():
        pae_logits = F.linear(F.layer_norm(zf, (zf.shape[-1],)) * g("pae_ln.weight") + bias("pae_ln.bias"),
                              g("linear_no_bias_pae.weight"))
        pde = F.linear(F.layer_norm(zf + zf.transpose(0, 1), (zf.shape[-1],)) * g("pde_ln.weight") + bias("pde_ln.bias"),
                       g("linear_no_bias_pde.weight"))
        a2t = feats["atom_to_token_idx"].long(); a2ta = feats["atom_to_tokatom_idx"].long()
        a = s_single[a2t]
        aln = F.layer_norm(a, (384,)) * g("plddt_ln.weight") + bias("plddt_ln.bias")
        logits = torch.einsum("nc,ncb->nb", aln, g("plddt_weight")[a2ta])
        return pae_logits, pde, logits
    t["heads_host"] = _time(heads, dev, warm=0, reps=5) * 1000
    pae_logits, pde_logits, plddt_logits = heads()

    # full confidence() for the total + the reference conf values
    def full():
        return ch.confidence(s_inputs, s_trunk, z_trunk, coords, feats)
    t["total_confidence"] = _time(full, dev, warm=1, reps=5) * 1000
    conf = full()
    return t, conf


def pad_to(s_inputs, s_trunk, z_trunk, coords, feats, Np):
    """Pad the real target's token/coord tensors to Np (repeat-block padding so
    the distance distribution stays realistic). Confidence requires the
    distogram-mask atom count == NT, so we pad to Np atoms with one atom per
    token (mask all ones, atom_to_token_idx = arange(Np)). The per-atom-type
    plddt lookup (atom_to_tokatom_idx) is tiled from the real target so its
    values stay in [0, n_tokatom)."""
    N = s_trunk.shape[0]
    if Np <= N:
        return s_inputs, s_trunk, z_trunk, coords, feats

    def rp(t, n):
        reps = (n + t.shape[0] - 1) // t.shape[0]
        return t.repeat((reps,) + (1,) * (t.dim() - 1))[:n].contiguous()
    s2 = rp(s_trunk, Np); si2 = rp(s_inputs, Np)
    z2 = rp(z_trunk.reshape(N, N, -1), Np).repeat(1, Np, 1)[:Np, :Np].contiguous()
    c2 = rp(coords, Np)
    a2t_orig = feats["atom_to_token_idx"].long()
    a2ta_orig = feats["atom_to_tokatom_idx"].long()
    f2 = dict(feats)
    f2["atom_to_token_idx"] = torch.arange(Np, dtype=torch.long)
    f2["atom_to_tokatom_idx"] = rp(a2ta_orig, Np)
    f2["distogram_rep_atom_mask"] = torch.ones(Np, dtype=torch.float32)
    f2["asym_id"] = rp(feats["asym_id"].reshape(-1), Np) if feats.get("asym_id") is not None else None
    return si2, s2, z2, c2, f2


def main():
    feats, Fraw, gold = _load_feats()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
    NT = int(feats["atom_to_token_idx"].max()) + 1
    print(f"[prof] cached target: NT={NT} tokens, N_atom={feats['ref_pos'].shape[0]}", flush=True)

    # 1) real e2e fold (n_step small) with confidence, timed
    _sync(dev); t0 = time.time()
    coords, conf = model.fold(feats, n_step=10, n_sample=1, seed=0, return_confidence=True)
    _sync(dev); t_fold = time.time() - t0
    print(f"[prof] e2e fold (n_step=10, n_sample=1): {t_fold*1000:.1f} ms; "
          f"plddt={conf['plddt']:.4f} ptm={conf['ptm']} iptm={conf['iptm']}", flush=True)

    # recover the real s_trunk / z_trunk / coords the fold used (re-derive via the trunk)
    s_inputs = model.fold.__func__  # not used; instead re-run trunk quickly via the cached inputs
    # Re-run just enough to get s_trunk/z_trunk/coords for the confidence split: use the fold's
    # returned coords + a fresh trunk pass (trunk is deterministic, no seed).
    fi = model._atom_feat_inputs(feats)
    s_inputs_tt = model.input_aae(
        model._tt(feats["ref_pos"]), model._tt(fi["ref_charge_asinh"]), model._tt(feats["ref_mask"].reshape(fi["N"], 1)),
        model._tt(fi["f_in"]), model._tt(fi["d"]), model._tt(fi["v"]), model._tt(fi["invd"]), fi["mt"], model._tt(
            (fi["S"].t() / (fi["S"].t().sum(-1, keepdim=True) + 1e-6))),
        model._tt(feats["restype"]), model._tt(feats["profile"]),
        model._tt(feats["deletion_mean"].reshape(-1, 1) if feats["deletion_mean"].dim() == 1 else feats["deletion_mean"]))
    s_inputs_h = model._to_host(s_inputs_tt)[:fi["NT"]]
    relp = feats["relp"] if "relp" in feats else model._generate_relp(feats)
    s_trunk_tt, z_tt = model.trunk(feats, s_inputs_h, relp, feats["token_bonds"], n_cycles=model.trunk.N_CYCLES)
    s_trunk = model._to_host(s_trunk_tt, (fi["NT"], s_trunk_tt.shape[-1]))
    z_trunk = model._to_host(z_tt, (fi["NT"], fi["NT"], model.trunk.C_Z))
    coords_h = coords[0]
    print(f"[prof] re-derived s_trunk {tuple(s_trunk.shape)} z_trunk {tuple(z_trunk.shape)} "
          f"coords {tuple(coords_h.shape)}", flush=True)

    ch = model.confidence_head

    # 2) real-target confidence split
    t_real, conf_real = confidence_split(ch, s_inputs_h, s_trunk, z_trunk, coords_h, feats, dev)
    print(f"[prof] confidence split @NT={fi['NT']} (ms): " + " ".join(f"{k}={v:.2f}" for k, v in t_real.items()), flush=True)
    print(f"[prof]   host_fraction(zembed+upload+download+heads) = "
          f"{(t_real['zembed_host']+t_real['upload_z']+t_real['upload_s']+t_real['download_dev']+t_real['heads_host']):.2f} ms "
          f"of total {t_real['total_confidence']:.2f} ms", flush=True)

    # 3) padded-N split (characterize scaling; values are padded, not a real fold)
    for Np in (128, 256):
        si2, s2, z2, c2, f2 = pad_to(s_inputs_h, s_trunk, z_trunk, coords_h, feats, Np)
        t_p, _ = confidence_split(ch, si2, s2, z2, c2, f2, dev)
        host = t_p["zembed_host"] + t_p["upload_z"] + t_p["upload_s"] + t_p["download_dev"] + t_p["heads_host"]
        print(f"[prof] confidence split @Np={Np} (ms, padded): " + " ".join(f"{k}={v:.2f}" for k, v in t_p.items()), flush=True)
        print(f"[prof]   host_fraction={host:.2f} ms of total {t_p['total_confidence']:.2f} ms "
              f"(device pairformer {t_p['pairformer_dev']:.2f} ms)", flush=True)

    print("PROF_DONE", flush=True)


if __name__ == "__main__":
    main()

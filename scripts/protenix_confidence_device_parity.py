"""Parity + timing for the device-resident Protenix-v2 confidence path.

Compares ConfidenceHead.confidence_device (z-embed + heads on device, z_base
resident) vs ConfidenceHead.confidence (host z-embed + host heads, the existing
validated path: pae/pde PCC 1.0 and plddt PCC ~0.93 vs the real v2 reference),
and both vs the cached reference pred (golden pae/pde/plddt), for:
  - pae  (token-token expected distance, (N,N))
  - pde  (token-token expected distance, (N,N))
  - plddt (per-atom plddt, (N_atom,))

The risky head is plddt: the host path is already PCC ~0.93 vs the reference
(the per-atom einsum over a 50-bin low-dynamic-range output is bf16-sensitive).
Moving the einsum to device bf16 can regress it. The device path is gated behind
TT_PROTENIX_CONF_DEVICE=1 and OFF by default; this harness enables it explicitly
and reports the device-vs-host and device-vs-gold PCC so the default can be set
on real evidence. Also reports warm device-path vs host-path wall-clock.

Reuses the one cached real target (gold NT=38 fold) plus a padded-N sweep so the
heads are stressed at the N where the device port is meant to pay off (N=256).
"""
import os, sys, time
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
os.environ["TT_PROTENIX_CONF_DEVICE"] = "1"   # exercise the device path explicitly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle, torch, torch.nn.functional as F, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix

CKPT = os.environ.get("PROTENIX_CKPT", "/home/moritz/.boltz/protenix-v2.pt")
IFE = os.environ.get("PROTENIX_IFE", "/home/moritz/protenix_ife_gold.pkl")
TG = os.environ.get("PROTENIX_TG", "/home/moritz/protenix_trunkin_gold.pkl")
REF = os.environ.get("PROTENIX_REF", "/home/moritz/protenix_ref_out.pkl")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    if a.numel() < 2:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm() + 1e-12))


def _expected(logits, max_a=32.0):
    nb = logits.shape[-1]
    centers = (torch.arange(nb, dtype=torch.float32) + 0.5) * (max_a / nb)
    return (torch.softmax(logits, -1) * centers).sum(-1)


def _plddt_atom(logits):
    nb = logits.shape[-1]
    return (torch.softmax(logits, -1) * ((torch.arange(nb, dtype=torch.float32) + 0.5) / nb)).sum(-1)


def load_feats():
    ife = pickle.load(open(IFE, "rb")); F = ife["feat"]
    d = pickle.load(open(REF, "rb"))
    tfeat = d["intermediates"]["template_embedder"]["in"][0]
    tg = pickle.load(open(TG, "rb"))
    rfeat = d.get("feat", {})
    feats = {
        "ref_pos": F["ref_pos"], "ref_charge": F["ref_charge"], "ref_mask": F["ref_mask"],
        "ref_element": F["ref_element"], "ref_atom_name_chars": F["ref_atom_name_chars"],
        "d_lm": F["d_lm"], "v_lm": F["v_lm"], "atom_to_token_idx": F["atom_to_token_idx"],
        "restype": F["restype"], "profile": F["profile"], "deletion_mean": F["deletion_mean"],
        "mask_trunked": ife["mask_trunked"], "relp": tg["relp"], "token_bonds": tg["token_bonds"],
        "template_aatype": tfeat["template_aatype"], "template_distogram": tfeat["template_distogram"],
        "template_pseudo_beta_mask": tfeat["template_pseudo_beta_mask"],
        "template_unit_vector": tfeat["template_unit_vector"],
        "template_backbone_frame_mask": tfeat["template_backbone_frame_mask"],
        "msa": tfeat["msa"], "has_deletion": tfeat["has_deletion"],
        "deletion_value": tfeat["deletion_value"], "asym_id": tfeat["asym_id"],
        "distogram_rep_atom_mask": rfeat["distogram_rep_atom_mask"],
        "atom_to_tokatom_idx": rfeat["atom_to_tokatom_idx"],
        "ref_space_uid": rfeat.get("ref_space_uid", torch.zeros(F["ref_pos"].shape[0], dtype=torch.long)),
    }
    return feats, d.get("pred", {})


def pad_to(s_inputs, s_trunk, z_trunk, coords, feats, Np):
    N = s_trunk.shape[0]
    if Np <= N:
        return s_inputs, s_trunk, z_trunk, coords, feats

    def rp(t, n):
        reps = (n + t.shape[0] - 1) // t.shape[0]
        return t.repeat((reps,) + (1,) * (t.dim() - 1))[:n].contiguous()
    s2 = rp(s_trunk, Np); si2 = rp(s_inputs, Np)
    z2 = rp(z_trunk.reshape(N, N, -1), Np).repeat(1, Np, 1)[:Np, :Np].contiguous()
    c2 = rp(coords, Np)
    a2t = feats["atom_to_token_idx"].long(); a2ta = feats["atom_to_tokatom_idx"].long()
    f2 = dict(feats)
    f2["atom_to_token_idx"] = torch.arange(Np, dtype=torch.long)
    f2["atom_to_tokatom_idx"] = rp(a2ta, Np)
    f2["distogram_rep_atom_mask"] = torch.ones(Np, dtype=torch.float32)
    f2["asym_id"] = rp(feats["asym_id"].reshape(-1), Np) if feats.get("asym_id") is not None else None
    return si2, s2, z2, c2, f2


def time_fn(fn, dev, warm=2, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    t = time.time()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.time() - t) / reps * 1000


def main():
    feats, gold = load_feats()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
    ch = model.confidence_head
    print("device_confidence_enabled:", ch.device_confidence_enabled(), flush=True)

    # real fold -> s_trunk, z_trunk (device), coords
    coords, _ = model.fold(feats, n_step=10, n_sample=1, seed=0, return_confidence=True)
    fi = model._atom_feat_inputs(feats)
    s_inputs_tt = model.input_aae(
        model._tt(feats["ref_pos"]), model._tt(fi["ref_charge_asinh"]), model._tt(feats["ref_mask"].reshape(fi["N"], 1)),
        model._tt(fi["f_in"]), model._tt(fi["d"]), model._tt(fi["v"]), model._tt(fi["invd"]), fi["mt"],
        model._tt((fi["S"].t() / (fi["S"].t().sum(-1, keepdim=True) + 1e-6))),
        model._tt(feats["restype"]), model._tt(feats["profile"]),
        model._tt(feats["deletion_mean"].reshape(-1, 1) if feats["deletion_mean"].dim() == 1 else feats["deletion_mean"]))
    s_inputs = model._to_host(s_inputs_tt)[:fi["NT"]]
    relp = feats["relp"] if "relp" in feats else model._generate_relp(feats)
    s_trunk_tt, z_tt = model.trunk(feats, s_inputs, relp, feats["token_bonds"], n_cycles=model.trunk.N_CYCLES)
    s_trunk = model._to_host(s_trunk_tt, (fi["NT"], s_trunk_tt.shape[-1]))
    z_trunk = model._to_host(z_tt, (fi["NT"], fi["NT"], model.trunk.C_Z))
    coords_h = coords[0]

    # golden per-atom plddt / token pae / pde (expected values)
    gold_pae = _expected(gold["pae"].float().reshape(gold["pae"].shape[-3:] if gold["pae"].dim() == 4 else gold["pae"].shape))
    gold_pde = _expected(gold["pde"].float().reshape(gold["pde"].shape[-3:] if gold["pde"].dim() == 4 else gold["pde"].shape))
    gold_plddt = _plddt_atom(gold["plddt"].float().reshape(-1, gold["plddt"].shape[-1]))

    def report(tag, si, st, zt_dev, zt_host, c, ft, g_pae=None, g_pde=None, g_plddt=None):
        ch.__dict__.pop("_dev_res", None)   # force re-resident per N (different z)
        conf_h = ch.confidence(si, st, zt_host, c, ft)
        conf_d = ch.confidence_device(si, st, zt_dev, c, ft)
        pae_h, pde_h, pl_h = conf_h["pae"], conf_h["pde"], conf_h["plddt_atom"]
        pae_d, pde_d, pl_d = conf_d["pae"], conf_d["pde"], conf_d["plddt_atom"]
        print(f"[{tag}] PCC device-vs-host: pae={_pcc(pae_d, pae_h):.4f} "
              f"pde={_pcc(pde_d, pde_h):.4f} plddt={_pcc(pl_d, pl_h):.4f}", flush=True)
        if g_pae is not None:
            print(f"[{tag}] PCC host-vs-gold:   pae={_pcc(pae_h, g_pae):.4f} "
                  f"pde={_pcc(pde_h, g_pde):.4f} plddt={_pcc(pl_h, g_plddt):.4f}", flush=True)
            print(f"[{tag}] PCC dev-vs-gold:    pae={_pcc(pae_d, g_pae):.4f} "
                  f"pde={_pcc(pde_d, g_pde):.4f} plddt={_pcc(pl_d, g_plddt):.4f}", flush=True)
        # timing (warm)
        th = time_fn(lambda: ch.confidence(si, st, zt_host, c, ft), dev)
        td = time_fn(lambda: ch.confidence_device(si, st, zt_dev, c, ft), dev)
        print(f"[{tag}] warm ms: host={th:.2f} device={td:.2f} (delta={th-td:+.2f})", flush=True)

    T = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zb_dev = ch.z_base_device(s_inputs, s_trunk, z_trunk)
    report("NT=38 real", s_inputs, s_trunk, zb_dev, z_trunk, coords_h, feats, gold_pae, gold_pde, gold_plddt)

    for Np in (128, 256):
        si2, s2, z2, c2, f2 = pad_to(s_inputs, s_trunk, z_trunk, coords_h, feats, Np)
        z2_dev = ch.z_base_device(si2, s2, z2)
        report(f"Np={Np} padded", si2, s2, z2_dev, z2, c2, f2)

    print("PARITY_DONE", flush=True)


if __name__ == "__main__":
    main()

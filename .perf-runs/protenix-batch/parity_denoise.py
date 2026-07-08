import os, sys, pickle
os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
os.environ['TT_PROTENIX_DBG_COND'] = '1'
sys.path.insert(0, '/home/ttuser/tt-bio-dev')
import torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix

CKPT = '/home/ttuser/protenix_ckpt/protenix-v2.pt'
ife = pickle.load(open('/home/ttuser/protenix_ife_gold.pkl', 'rb'))
tg = pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl', 'rb'))
d = pickle.load(open('/home/ttuser/protenix_ref_out.pkl', 'rb'))
tfeat = d['intermediates']['template_embedder']['in'][0]
F = ife['feat']
feats = {
    'ref_pos': F['ref_pos'], 'ref_charge': F['ref_charge'], 'ref_mask': F['ref_mask'],
    'ref_element': F['ref_element'], 'ref_atom_name_chars': F['ref_atom_name_chars'],
    'd_lm': F['d_lm'], 'v_lm': F['v_lm'], 'atom_to_token_idx': F['atom_to_token_idx'],
    'restype': F['restype'], 'profile': F['profile'], 'deletion_mean': F['deletion_mean'],
    'mask_trunked': ife['mask_trunked'], 'relp': tg['relp'], 'token_bonds': tg['token_bonds'],
    'template_aatype': tfeat['template_aatype'], 'template_distogram': tfeat['template_distogram'],
    'template_pseudo_beta_mask': tfeat['template_pseudo_beta_mask'],
    'template_unit_vector': tfeat['template_unit_vector'],
    'template_backbone_frame_mask': tfeat['template_backbone_frame_mask'],
    'msa': tfeat['msa'], 'has_deletion': tfeat['has_deletion'], 'deletion_value': tfeat['deletion_value'],
    'asym_id': tfeat['asym_id'],
}
FAST = '--fast' in sys.argv
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
model = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
model._fast = FAST
# populate cond via one short serial fold
os.environ['TT_PROTENIX_NOBATCH'] = '1'
_ = model.fold(feats, n_step=3, n_sample=1, seed=0)
cond = model._dbg_cond
N = cond['c_l'].shape[0]
print(f"cond ready N={N} fast={FAST}", flush=True)

def pcc(a, b):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / (((a - a.mean()).pow(2).sum().sqrt()) * ((b - b.mean()).pow(2).sum().sqrt()) + 1e-12))

torch.manual_seed(0)
for t_hat_v in [8.0]:
    th = torch.tensor([t_hat_v], dtype=torch.float32)
    x1 = torch.randn(1, N, 3)
    dn_s = model.diffusion.denoise(x1, th, cond).float()          # (1,N,3)
    dn_s2 = model.diffusion.denoise(x1, th, cond).float()         # determinism floor
    print(f"[serial x2] maxdiff={float((dn_s-dn_s2).abs().max()):.3e} PCC={pcc(dn_s, dn_s2):.6f}", flush=True)
    # B=1 batched vs serial
    dn_b1 = model.diffusion.denoise_batched(x1.contiguous(), th, cond, 1).float()
    print(f"[B=1] maxdiff={float((dn_b1[0]-dn_s[0]).abs().max()):.3e} PCC={pcc(dn_b1[0], dn_s[0]):.6f}", flush=True)
    # B=3 all-identical: samples must equal each other AND serial
    B = 3
    xB = x1.repeat(B, 1, 1).contiguous()
    dn_b = model.diffusion.denoise_batched(xB, th, cond, B).float()  # (B,N,3)
    for k in range(B):
        print(f"[B=3] s{k} vs serial: maxdiff={float((dn_b[k]-dn_s[0]).abs().max()):.3e} PCC={pcc(dn_b[k], dn_s[0]):.6f}", flush=True)
    print(f"[B=3] s0 vs s1: maxdiff={float((dn_b[0]-dn_b[1]).abs().max()):.3e}", flush=True)
    print(f"[B=3] s0 vs s2: maxdiff={float((dn_b[0]-dn_b[2]).abs().max()):.3e}", flush=True)
print('PARITY_DONE', flush=True)

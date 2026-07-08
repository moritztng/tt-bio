import os, sys, pickle
os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
os.environ['TT_PROTENIX_DBG_COND'] = '1'
os.environ['TT_PROTENIX_TRACE_REGION'] = str(1024 * 1024 * 1024)   # open device with trace region
sys.path.insert(0, '/home/ttuser/tt-bio-dev')
import torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix, edm_sample
ife = pickle.load(open('/home/ttuser/protenix_ife_gold.pkl', 'rb'))
tg = pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl', 'rb'))
d = pickle.load(open('/home/ttuser/protenix_ref_out.pkl', 'rb')); tfeat = d['intermediates']['template_embedder']['in'][0]
traj = pickle.load(open('/home/ttuser/protenix_traj.pkl', 'rb'))
F = ife['feat']
feats = {'ref_pos': F['ref_pos'], 'ref_charge': F['ref_charge'], 'ref_mask': F['ref_mask'], 'ref_element': F['ref_element'],
    'ref_atom_name_chars': F['ref_atom_name_chars'], 'd_lm': F['d_lm'], 'v_lm': F['v_lm'], 'atom_to_token_idx': F['atom_to_token_idx'],
    'restype': F['restype'], 'profile': F['profile'], 'deletion_mean': F['deletion_mean'], 'mask_trunked': ife['mask_trunked'],
    'relp': tg['relp'], 'token_bonds': tg['token_bonds'], 'template_aatype': tfeat['template_aatype'],
    'template_distogram': tfeat['template_distogram'], 'template_pseudo_beta_mask': tfeat['template_pseudo_beta_mask'],
    'template_unit_vector': tfeat['template_unit_vector'], 'template_backbone_frame_mask': tfeat['template_backbone_frame_mask'],
    'msa': tfeat['msa'], 'has_deletion': tfeat['has_deletion'], 'deletion_value': tfeat['deletion_value'], 'asym_id': tfeat['asym_id']}
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
model = Protenix.load_from_checkpoint('/home/ttuser/protenix_ckpt/protenix-v2.pt', compute_kernel_config=ckc, device=dev)
model._fast = True
os.environ['TT_PROTENIX_TRACE'] = ''
_ = model.fold(feats, n_step=3, n_sample=1, seed=0)   # populate cond
cond = model._dbg_cond
N = cond['c_l'].shape[0]
NS = 200
# untraced
os.environ.pop('TT_PROTENIX_TRACE', None)
xu = edm_sample(model.diffusion, cond, N, n_step=NS, seed=0)[0].float()
# traced (fresh trace)
model.diffusion._trace = None
os.environ['TT_PROTENIX_TRACE'] = '1'
xt = edm_sample(model.diffusion, cond, N, n_step=NS, seed=0)[0].float()
def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0); H = Pc.t() @ Qc
    U, _, Vt = torch.linalg.svd(H); Dm = torch.diag(torch.tensor([1., 1., torch.sign(torch.det(Vt.t() @ U.t()))]))
    R = Vt.t() @ Dm @ U.t(); return float(((Pc @ R.t()) - Qc).pow(2).sum(-1).mean().sqrt())
print(f"N={N} n_step={NS}")
print(f"trace-vs-untraced (same seed) coord maxdiff={float((xu-xt).abs().max()):.3e}  Kabsch RMSD={kabsch(xu,xt):.4f} A")
ref = traj.get('final_coords')
if ref is not None:
    rf = ref.float().reshape(-1, 3)[:N]
    print(f"untraced vs ref RMSD={kabsch(xu,rf):.4f} A ; trace vs ref RMSD={kabsch(xt,rf):.4f} A")
print("ACC_DONE")

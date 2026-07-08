import os, sys, pickle
os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
os.environ['TT_PROTENIX_DBG_COND'] = '1'; os.environ['TT_PROTENIX_NOBATCH'] = '1'
sys.path.insert(0, '/home/ttuser/tt-bio-dev')
import torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix
ife = pickle.load(open('/home/ttuser/protenix_ife_gold.pkl', 'rb'))
tg = pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl', 'rb'))
d = pickle.load(open('/home/ttuser/protenix_ref_out.pkl', 'rb')); tfeat = d['intermediates']['template_embedder']['in'][0]
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
_ = model.fold(feats, n_step=3, n_sample=1, seed=0)
cond = model._dbg_cond
NT = cond['s_inputs'].shape[0]
biases = cond['dit_block_biases']
adaln_a, apb, ctb_adaln, A, Cc = model.diffusion._dit[1]   # block 1 (the one that diverged)
bias = biases[1]
def to_t(x): return ttnn.to_torch(x).float()
def pcc(a, b):
    a=a.reshape(-1).double(); b=b.reshape(-1).double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).pow(2).sum().sqrt()*(b-b.mean()).pow(2).sum().sqrt()+1e-12))
torch.manual_seed(0)
a1 = ttnn.from_torch(torch.randn(1, NT, 768), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
b1 = adaln_a(a1, cond['ss_base'] if False else ttnn.from_torch(torch.randn(1,NT,cond['s_trunk'].shape[-1] if False else 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16))
# simpler: skip adaln; feed random b directly to apb at B=1 and B=3-identical
torch.manual_seed(1)
bb1 = ttnn.from_torch(torch.randn(1, NT, 768), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
o1 = to_t(apb(bb1, bias, bias_precomputed=True))               # (1,NT,768)
bb3 = ttnn.from_torch(to_t(bb1).repeat(3,1,1).contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
o3 = to_t(apb(bb3, bias, bias_precomputed=True))               # (3,NT,768)
print("apb hd=48: o3[0]-vs-o1 maxdiff", float((o3[0]-o1[0]).abs().max()), "PCC", round(pcc(o3[0],o1[0]),6))
print("apb hd=48: o3[0]-vs-o3[1] maxdiff", float((o3[0]-o3[1]).abs().max()))
print("MICRO_APB_DONE")

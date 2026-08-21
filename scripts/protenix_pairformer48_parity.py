import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
# tt-boltz2 only supplies protenix_reference, and it carries its own stale copy of tt_bio.
# Append, never insert(0): at the front it outranks PYTHONPATH and the script then scores
# /home/ttuser/tt-boltz2 instead of the checkout under test.
sys.path.append('/home/ttuser/tt-boltz2'); sys.path.append('/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from protenix_reference import remap_pairformer_block
from tt_bio.tenstorrent import get_device, Pairformer, accurate_softmax_site
import pathlib, tt_bio
_here = pathlib.Path(__file__).resolve().parent.parent
assert pathlib.Path(tt_bio.__file__).resolve().parent.parent == _here, \
    'tt_bio resolved to %s, not %s' % (tt_bio.__file__, _here)
ck_all=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck_all=ck_all.get('model',ck_all)
# count blocks + dims
import re
nb=1+max(int(re.search(r'pairformer_stack\.blocks\.(\d+)\.',k).group(1)) for k in ck_all if 'pairformer_stack.blocks.' in k and re.search(r'pairformer_stack\.blocks\.(\d+)\.',k))
b0={k[len('module.pairformer_stack.blocks.0.'):]:v for k,v in ck_all.items() if k.startswith('module.pairformer_stack.blocks.0.')}
c_z=b0["tri_mul_in.layer_norm_in.weight"].shape[0]; c_s=b0["single_transition.layernorm1.weight"].shape[0]
no_heads_pair=b0["tri_att_start.linear.weight"].shape[0]
c_hidden_pair_att=b0["tri_att_start.mha.linear_q.weight"].shape[0]//no_heads_pair
apb_nh=b0["attention_pair_bias.linear_nobias_z.weight"].shape[0]
print('n_blocks=%d c_z=%d c_s=%d no_heads_pair=%d c_hid_pair_att=%d apb_nh=%d'%(nb,c_z,c_s,no_heads_pair,c_hidden_pair_att,apb_nh),flush=True)
# build combined remapped sd under layers.{i}.
combined={}
for i in range(nb):
    pfx=f'module.pairformer_stack.blocks.{i}.'
    bsd={k[len(pfx):]:v for k,v in ck_all.items() if k.startswith(pfx)}
    for k,v in remap_pairformer_block(bsd).items(): combined[f'layers.{i}.{k}']=v
dev=get_device()
cfg=ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
# Construct exactly what protenix.py:2194 ships, selector and all, so this anchor scores the
# shipped trunk and TT_BIO_ACCURATE_SOFTMAX_AB=-all gives the honest off arm. Passing nothing
# would take Pairformer's own accurate_softmax=False and score OFF in both arms.
_acc = accurate_softmax_site('protenix.trunk', default=True)
print('accurate_softmax=%s' % _acc, flush=True)
pf=Pairformer(nb, c_hidden_pair_att, no_heads_pair, c_s//apb_nh, apb_nh, True, combined, cfg,
              accurate_softmax=_acc)
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
io=d['intermediates']['pairformer_stack']; (s_in,z_in)=io['in']; (s_out,z_out)=io['out']
s_in,z_in,s_out,z_out=[x.float() for x in (s_in,z_in,s_out,z_out)]
ft=lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
so,zo=pf(ft(s_in.unsqueeze(0)), ft(z_in.unsqueeze(0)))
so=torch.Tensor(ttnn.to_torch(so)).float().reshape(s_out.shape); zo=torch.Tensor(ttnn.to_torch(zo)).float().reshape(z_out.shape)
def pcc(u,v):
    u=u.flatten().double(); v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
_ps, _pz = pcc(so, s_out), pcc(zo, z_out)
# 7 dp, and 1-PCC alongside: the A/B delta on this anchor is in the 5th decimal, which
# %.5f cannot resolve.
print('Pairformer x%d  s PCC %.7f  z PCC %.7f  (1-PCC: s %.3e  z %.3e)'
      % (nb, _ps, _pz, 1 - _ps, 1 - _pz), flush=True)

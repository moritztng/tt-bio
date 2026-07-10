# Device-side counterpart to protenix_hemo_ref_cycles.py: feeds the REFERENCE-computed
# s_inputs/relp/token_bonds/feat (hemoglobin tetramer, 574 tokens, use_msa=False,
# use_template=False) into tt_bio.protenix.Trunk at n_cycles=1..K and compares against the
# reference's own per-cycle (s,z) capture. Answers: does on-device trunk recycling drift
# more at hemoglobin's size (574 tok, tetramer) than at the validated 38-tok case (PCC
# s=0.991/z=0.990 @ 10 cycles, scripts/protenix_trunk_assembly.py)?
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES', '0'); os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Trunk

d = pickle.load(open('/home/ttuser/protenix_hemo_ref_cycles.pkl', 'rb'))
cycles = d['cycles']          # list of (s,z) torch tensors, index k-1 = after cycle k
s_inputs = d['s_inputs'].float()
relp = d['relp'].float()
token_bonds = d['token_bonds'].float()
feat = d['feat_small']
N = d['N_token']
print('N_token=%d  #cycles captured=%d'%(N, len(cycles)), flush=True)

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
ck = torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt', map_location='cpu', weights_only=True)
ck = ck.get('model', ck)
sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ck.items()}
trunk = Trunk(sd, ckc)

def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))

print(f"{'cycle':>6}{'s_PCC':>10}{'z_PCC':>10}{'s_absmean(ref/dev)':>22}{'z_absmean(ref/dev)':>22}", flush=True)
for k in range(1, len(cycles) + 1):
    s_ref, z_ref = cycles[k - 1]
    s_dev, z_dev = trunk(feat, s_inputs, relp, token_bonds, n_cycles=k)
    s_dev_h = torch.Tensor(ttnn.to_torch(s_dev)).float().reshape(s_ref.shape)
    z_dev_h = torch.Tensor(ttnn.to_torch(z_dev)).float().reshape(z_ref.shape)
    ps = pcc(s_dev_h, s_ref); pz = pcc(z_dev_h, z_ref)
    print(f"{k:>6}{ps:>10.5f}{pz:>10.5f}"
          f"{f'{s_ref.abs().mean():.3e}/{s_dev_h.abs().mean():.3e}':>22}"
          f"{f'{z_ref.abs().mean():.3e}/{z_dev_h.abs().mean():.3e}':>22}", flush=True)
print('HEMO_DEVICE_TRUNK_PCC_DONE', flush=True)

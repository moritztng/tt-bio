# Reference Protenix-v2 WITH the real precomputed MSA (examples/msa/seq2.a3m),
# folding the full 134-res 7ROA construct (the a3m query row), proper sampling,
# then Kabsch CA-RMSD vs 7ROA matched by label_seq (pred position == entity
# label_seq == crystal label_seq, so resolved residues map 1:1). This is the
# control that tells us whether the target folds well GIVEN an MSA -- isolating
# "no-MSA regime" as the root cause of the ~5-10A no-MSA results.
import sys, types, numbers, os, pickle, json
os.environ.setdefault('PROTENIX_ROOT_DIR', '/home/ttuser')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tt-bio repo root
sys.path.insert(0, os.environ.get('PROTENIX_SRC', '/tmp/protenix-src'))
import numpy as np
import torch, torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
torch.set_grad_enabled(False)

A3M = os.environ.get('A3M', os.path.join(_REPO, 'examples/msa/seq2.a3m'))
N_STEP  = int(os.environ.get('N_STEP', '200'))
N_SAMPLE = int(os.environ.get('N_SAMPLE', '5'))
SEED = int(os.environ.get('SEED', '42'))
GT_CIF = os.environ.get('GT_CIF', os.path.join(_REPO, 'examples/ground_truth_structures/prot.cif'))
OUT_CIF = os.environ.get('OUT_CIF', '/tmp/protenix_ref_prot_msa_best.cif')
INPUT_JSON = '/tmp/protenix_prot_msa.json'

# input sequence = a3m query row (full construct), X(=MSE) -> M
with open(A3M) as f:
    lines = f.read().splitlines()
query = lines[1].strip().replace('X', 'M')
print('a3m query len=%d (full construct)'%len(query), flush=True)
with open(INPUT_JSON, 'w') as f:
    json.dump([{"sequences": [{"proteinChain": {"sequence": query, "count": 1,
                "unpairedMsaPath": A3M}}],
                "modelSeeds": [SEED], "name": "prot_msa"}], f)

stub = types.ModuleType('protenix.model.layer_norm.layer_norm')
class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, create_scale=True, create_offset=True, eps=1e-5, **kw):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape=(normalized_shape,)
        self.normalized_shape=tuple(normalized_shape); self.eps=eps
        self.weight = Parameter(torch.ones(*normalized_shape)) if create_scale else None
        self.bias = Parameter(torch.zeros(*normalized_shape)) if create_offset else None
    def forward(self, x):
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if self.weight is not None: x = x*self.weight
        if self.bias is not None: x = x+self.bias
        return x
stub.FusedLayerNorm = FusedLayerNorm
sys.modules['protenix.model.layer_norm.layer_norm'] = stub

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from protenix.model.protenix import Protenix
from protenix.data.inference.infer_dataloader import get_inference_dataloader

base = {**configs_base, **{"data": data_configs}, **inference_configs}
def du(d,u):
    for k,v in u.items():
        d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base, model_configs['protenix-v2'])
base['input_json_path'] = INPUT_JSON
base['dump_dir'] = '/tmp/protenix_msa_out'
base['use_msa'] = True
base['use_template'] = False
base['use_seeds_in_json'] = True
base['triangle_multiplicative'] = 'torch'
base['triangle_attention'] = 'torch'
if isinstance(base.get('esm'), dict): base['esm']['enable'] = False
base['sample_diffusion']['N_step'] = N_STEP
base['sample_diffusion']['N_sample'] = N_SAMPLE
cfg = parse_configs(base, fill_required_with_null=True)

print('building v2 model (use_msa=True)...', flush=True)
m = Protenix(cfg).eval()
ck = torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt', map_location='cpu', weights_only=True)
ck = ck.get('model', ck)
sd = {k[len('module.'):] if k.startswith('module.') else k: v for k,v in ck.items()}
miss, unexp = m.load_state_dict(sd, strict=False)
print('V2 load: missing=%d unexpected=%d'%(len(miss), len(unexp)), flush=True)

dl = get_inference_dataloader(configs=cfg)
data, atom_array, err = next(iter(dl))[0]
if err: print('DATA ERROR:', err, flush=True)
feat = data['input_feature_dict']
print('N_token=%d N_atom=%d N_msa=%d'%(int(data['N_token']), int(data['N_atom']), int(data['N_msa'])), flush=True)

print('running reference forward WITH MSA...', flush=True)
pred, _, _ = m(input_feature_dict=feat, label_full_dict=None, label_dict=None,
               mode='inference', mc_dropout_apply_rate=0.0)
coords = pred['coordinate']
scs = pred['summary_confidence']
rank = [float(s['ranking_score']) for s in scs]
plddts = [float(s['plddt']) for s in scs]; ptms = [float(s['ptm']) for s in scs]
best = int(np.argmax(rank))
print('per-sample ranking_score=%s'%['%.4f'%r for r in rank], flush=True)
print('BEST sample=%d ranking_score=%.4f plddt=%.2f ptm=%.4f'%(best,rank[best],plddts[best],ptms[best]), flush=True)
best_coords = coords[best].cpu().numpy().astype(np.float64)

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
aa = atom_array.copy()
assert aa.array_length() == best_coords.shape[0]
aa.coord = best_coords.astype(np.float32)
bcif = pdbx.CIFFile(); pdbx.set_structure(bcif, aa); bcif.write(OUT_CIF)
print('wrote best -> %s'%OUT_CIF, flush=True)

import gemmi
def ca_by_labelseq(path):
    st = gemmi.read_structure(path); st.setup_entities(); st.remove_alternative_conformations()
    out = {}
    for chain in st[0]:
        poly = chain.get_polymer()
        for i,res in enumerate(poly):
            ca = res.find_atom("CA","*")
            if ca is None: continue
            ls = res.label_seq if res.label_seq is not None else (i+1)
            out.setdefault(chain.name, {})[int(ls)] = (ca.pos.x, ca.pos.y, ca.pos.z)
    return out
pred_ca = ca_by_labelseq(OUT_CIF); gt_ca = ca_by_labelseq(GT_CIF)
print('pred label_seq n=%d range %s..%s'%(len(pred_ca['A']), min(pred_ca['A']), max(pred_ca['A'])), flush=True)
print('gt   label_seq n=%d range %s..%s'%(len(gt_ca['A']), min(gt_ca['A']), max(gt_ca['A'])), flush=True)
def rmsd(pd, gd):
    keys = sorted(set(pd)&set(gd))
    if len(keys)<3: return None,0
    P=[gemmi.Position(*pd[k]) for k in keys]; Q=[gemmi.Position(*gd[k]) for k in keys]
    return gemmi.superpose_positions(P,Q).rmsd, len(keys)
best_r, best_n, best_pair = None, 0, None
for pc,pd in pred_ca.items():
    for gc,gd in gt_ca.items():
        r,n = rmsd(pd,gd)
        if r is None: continue
        if best_r is None or r<best_r: best_r,best_n,best_pair=r,n,(pc,gc)
print('='*60, flush=True)
print('REFERENCE Protenix-v2 WITH MSA (seq2.a3m), full 134-res construct', flush=True)
print('sampling: N_step=%d N_sample=%d best=%d  pLDDT=%.2f pTM=%.4f'%(N_STEP,N_SAMPLE,best,plddts[best],ptms[best]), flush=True)
print('label_seq-matched n=%d  GROUND-TRUTH CA-RMSD vs 7ROA = %.4f A'%(best_n, best_r), flush=True)
print('='*60, flush=True)
print('REF_MSA_RMSD_DONE', flush=True)

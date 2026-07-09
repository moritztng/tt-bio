# Reference Protenix-v2 (official ByteDance torch impl) end-to-end fold of
# examples/prot.yaml chain A (117-res, PDB 7ROA), CPU, PROPER sampling settings
# (N_step=200, N_sample=5, best-by-ranking_score), then Kabsch CA-RMSD vs the
# 7ROA ground truth. Same OFFLINE feature regime as tt-bio's device path
# (use_msa=False, use_template=False, esm disabled) so the ONLY variable vs the
# tt-bio ~10A result is the model implementation + sampling depth.
#
# Run in the py3.11 protenix reference venv; needs CCD in ~/common/.
import sys, types, numbers, os, pickle, json
os.environ.setdefault('PROTENIX_ROOT_DIR', '/home/ttuser')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tt-bio repo root
sys.path.insert(0, os.environ.get('PROTENIX_SRC', '/tmp/protenix-src'))
import numpy as np
import torch, torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
torch.set_grad_enabled(False)

SEQ = "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"
N_STEP  = int(os.environ.get('N_STEP', '200'))
N_SAMPLE = int(os.environ.get('N_SAMPLE', '5'))
SEED = int(os.environ.get('SEED', '42'))
GT_CIF = os.environ.get('GT_CIF', os.path.join(_REPO, 'examples/ground_truth_structures/prot.cif'))
OUT_CIF = os.environ.get('OUT_CIF', '/tmp/protenix_ref_prot_best.cif')
INPUT_JSON = '/tmp/protenix_prot_refcheck.json'

with open(INPUT_JSON, 'w') as f:
    json.dump([{"sequences": [{"proteinChain": {"sequence": SEQ, "count": 1}}],
                "modelSeeds": [SEED], "name": "prot"}], f)

# --- stub CUDA FusedLayerNorm with a torch equivalent (CPU-only box) ---
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
base['dump_dir'] = '/tmp/protenix_refcheck_out'
base['use_msa'] = False
base['use_template'] = False
base['use_seeds_in_json'] = True
base['triangle_multiplicative'] = 'torch'
base['triangle_attention'] = 'torch'
if isinstance(base.get('esm'), dict): base['esm']['enable'] = False
base['sample_diffusion']['N_step'] = N_STEP
base['sample_diffusion']['N_sample'] = N_SAMPLE
cfg = parse_configs(base, fill_required_with_null=True)

print('building v2 model... N_step=%d N_sample=%d seed=%d'%(N_STEP,N_SAMPLE,SEED), flush=True)
m = Protenix(cfg).eval()
ck = torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt', map_location='cpu', weights_only=True)
ck = ck.get('model', ck)
sd = {k[len('module.'):] if k.startswith('module.') else k: v for k,v in ck.items()}
miss, unexp = m.load_state_dict(sd, strict=False)
print('V2 load: missing=%d unexpected=%d'%(len(miss), len(unexp)), flush=True)

dl = get_inference_dataloader(configs=cfg)
batch = next(iter(dl))
data, atom_array, err = batch[0]
if err: print('DATA ERROR:', err, flush=True)
feat = data['input_feature_dict']
print('N_token=%d N_atom=%d N_msa=%d'%(int(data['N_token']), int(data['N_atom']), int(data['N_msa'])), flush=True)

print('running reference forward (this is the slow part)...', flush=True)
pred, _, _ = m(input_feature_dict=feat, label_full_dict=None, label_dict=None,
               mode='inference', mc_dropout_apply_rate=0.0)

coords = pred['coordinate']          # (N_sample, N_atom, 3)
scs = pred['summary_confidence']     # list of per-sample dicts
rank = [float(s['ranking_score']) for s in scs]
plddts = [float(s['plddt']) for s in scs]
ptms = [float(s['ptm']) for s in scs]
best = int(np.argmax(rank))
print('per-sample ranking_score=%s'%['%.4f'%r for r in rank], flush=True)
print('per-sample plddt=%s'%['%.2f'%p for p in plddts], flush=True)
print('per-sample ptm=%s'%['%.4f'%p for p in ptms], flush=True)
print('BEST sample=%d ranking_score=%.4f plddt=%.2f ptm=%.4f'%(best,rank[best],plddts[best],ptms[best]), flush=True)

best_coords = coords[best].cpu().numpy().astype(np.float64)  # (N_atom,3)

# write best sample to CIF via biotite atom_array
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
aa = atom_array.copy()
assert aa.array_length() == best_coords.shape[0], (aa.array_length(), best_coords.shape)
aa.coord = best_coords.astype(np.float32)
bcif = pdbx.CIFFile()
pdbx.set_structure(bcif, aa)
bcif.write(OUT_CIF)
print('wrote best predicted sample -> %s'%OUT_CIF, flush=True)

# ---- Kabsch CA-RMSD vs 7ROA ground truth (match by label_seq_id) ----
import gemmi
def ca_by_labelseq(path, want_chain=None):
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_alternative_conformations()
    out = {}
    model = st[0]
    for chain in model:
        poly = chain.get_polymer()
        for i, res in enumerate(poly):
            ca = res.find_atom("CA", "*")
            if ca is None: continue
            ls = res.label_seq if res.label_seq is not None else (i+1)
            out.setdefault(chain.name, {})[int(ls)] = (ca.pos.x, ca.pos.y, ca.pos.z)
    return out

pred_ca = ca_by_labelseq(OUT_CIF)
gt_ca = ca_by_labelseq(GT_CIF)
print('pred chains: %s'%{c:len(v) for c,v in pred_ca.items()}, flush=True)
print('gt   chains: %s'%{c:len(v) for c,v in gt_ca.items()}, flush=True)

# pick the single predicted protein chain and the gt chain giving lowest RMSD
def rmsd_between(pd, gd):
    keys = sorted(set(pd) & set(gd))
    if len(keys) < 3: return None, 0
    P = [gemmi.Position(*pd[k]) for k in keys]
    Q = [gemmi.Position(*gd[k]) for k in keys]
    sup = gemmi.superpose_positions(P, Q)
    return sup.rmsd, len(keys)

best_rmsd, best_n, best_pair = None, 0, None
for pc, pd in pred_ca.items():
    for gc, gd in gt_ca.items():
        r, n = rmsd_between(pd, gd)
        if r is None: continue
        if best_rmsd is None or r < best_rmsd:
            best_rmsd, best_n, best_pair = r, n, (pc, gc)

print('='*60, flush=True)
print('REFERENCE Protenix-v2 (official torch, offline/no-MSA regime)', flush=True)
print('sampling: N_step=%d N_sample=%d seed=%d best_sample=%d (by ranking_score)'%(N_STEP,N_SAMPLE,SEED,best), flush=True)
print('best-sample pLDDT=%.2f pTM=%.4f'%(plddts[best], ptms[best]), flush=True)
print('chain match pred %s -> gt %s, %d matched CA'%(best_pair[0], best_pair[1], best_n), flush=True)
print('GROUND-TRUTH CA-RMSD (Kabsch, vs 7ROA): %.4f A'%best_rmsd, flush=True)
print('='*60, flush=True)
print('REF_GT_RMSD_DONE', flush=True)

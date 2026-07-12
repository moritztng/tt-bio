import sys, types, numbers, os
os.environ.setdefault('PROTENIX_ROOT_DIR', '/home/ttuser')
sys.path.insert(0, os.environ.get('PROTENIX_SRC', '/tmp/protenix-src'))
import torch, torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
torch.set_grad_enabled(False)

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
from protenix.data.inference.infer_dataloader import get_inference_dataloader

base = {**configs_base, **{"data": data_configs}, **inference_configs}
def du(d,u):
    for k,v in u.items():
        d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base, model_configs['protenix-v2'])
base['input_json_path'] = '/home/ttuser/protenix_hemo.json'
base['dump_dir'] = '/home/ttuser/protenix_out_hemo_featcheck'
base['use_msa'] = False
base['use_template'] = False
base['use_seeds_in_json'] = True
if isinstance(base.get('esm'), dict): base['esm']['enable'] = False
cfg = parse_configs(base, fill_required_with_null=True)

dl = get_inference_dataloader(configs=cfg)
for b in dl:
    data, atom_array, err = b[0]
    break
feat = data['input_feature_dict']
tkeys = [k for k in feat if 'template' in k]
print('template keys present:', tkeys)
for k in tkeys:
    v = feat[k]
    if torch.is_tensor(v):
        vf = v.float()
        print(f'  {k}: shape={tuple(v.shape)} dtype={v.dtype} nonzero={int((vf!=0).sum())} sum_abs={float(vf.abs().sum()):.4g}')
    else:
        print(f'  {k}: {type(v)} {v}')

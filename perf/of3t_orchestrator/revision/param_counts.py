import sys, torch
TREE = sys.argv[1]
sys.path.insert(0, TREE)
import openfold3
assert openfold3.__file__.startswith(TREE)
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3
torch.manual_seed(0)
m = OpenFold3(model_config)
named = list(m.named_parameters())
print(f"{sys.argv[2]}: parameters()={len(named)}  state_dict()={len(m.state_dict())}")

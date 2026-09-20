import sys, json, torch
TREE="/home/moritz/.coworker/scratch/of3t-reference/upstream"
sys.path.insert(0,TREE)
from openfold3.core.utils.atomize_utils import get_token_representative_atoms
b=torch.load("/home/moritz/.coworker/scratch/of3t-reference/bundle/batches/batch_step003.pt",map_location="cpu",weights_only=False)
gt=b["ground_truth"]
x=gt.get("atom_positions", gt.get("x_exp"))
if x is None:
    x=torch.zeros(b["atom_mask"].shape+(3,))
_, repr_mask = get_token_representative_atoms(batch=b, x=x, atom_mask=b["atom_mask"])
tok=b["token_mask"]
d=(repr_mask.float()-tok.float())
print(json.dumps({"repr_mask_sum":float(repr_mask.float().sum()),
                  "token_mask_sum":float(tok.float().sum()),
                  "n_differing_tokens":int((d!=0).sum()),
                  "identical":bool((d==0).all())}))

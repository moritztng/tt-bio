import glob, json, os, sys, torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p2")
from safetensors.torch import load_file
from tt_bio import tenstorrent
import tt_bio.esmfold2 as E

snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--biohub--ESMFold2/snapshots/*"))[0]
sd = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    for k, v in load_file(f).items():
        if k.startswith("msa_encoder."):
            sd[k[len("msa_encoder."):]] = v

B, L, M = 1, 128, 256
g = torch.Generator().manual_seed(0)
mk = lambda *s: torch.randn(*s, generator=g)
inputs = dict(x_pair=mk(B, L, L, 256), x_inputs=mk(B, L, 451),
    msa_oh=torch.nn.functional.one_hot(torch.randint(0, 33, (B, L, M), generator=g), 33).float(),
    has_deletion=mk(B, L, M).gt(0).float(), deletion_value=mk(B, L, M).abs(),
    msa_mask=torch.ones(B, L, M))

_orig = E._msa_row_blocks
def only(site):
    def patched(Ln, Mn):
        f = sys._getframe(1)
        cls = type(f.f_locals["self"]).__name__ if "self" in f.f_locals else f.f_code.co_name
        return _orig(Ln, Mn) if cls == site else None
    return patched

def run(patch):
    tenstorrent.SMALL_GRID_MSA_TILE_AREA = 8192 if patch else 0
    E._msa_row_blocks = patch or _orig
    mod = E.MSAEncoder(4, 8, 16); mod.load_state_dict(sd, strict=False)
    return torch.Tensor(mod(**inputs)).clone()

base = run(None)
for site in ("OuterProductMean", "MSAPairWeightedAveraging", "_msa_rowwise_residual"):
    out = run(only(site))
    d = (base - out).abs().max().item()
    print(f"{site:28s} equal={torch.equal(base, out)}  maxdiff={d}", flush=True)

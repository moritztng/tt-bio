"""Bit-exactness A/B for the MSA-encoder row blocking, on Blackhole.

Same weights, same inputs, one arm forced through the row-blocked path by
setting SMALL_GRID_MSA_TILE_AREA (inert on a big grid otherwise). Pass =
torch.equal on the returned pair tensor.
"""
import glob, json, os, sys, torch
sys.path.insert(0, "/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p2")
from safetensors.torch import load_file
from tt_bio import tenstorrent
import tt_bio.esmfold2 as E

snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--biohub--ESMFold2/snapshots/*"))[0]
cfg = json.load(open(os.path.join(snap, "config.json")))
msa_cfg = cfg["msa_encoder"] if "msa_encoder" in cfg else cfg["config"]["msa_encoder"]
n_layers = msa_cfg.get("n_layers", 4)
n_heads = msa_cfg.get("n_heads_msa", 8)
head_w = msa_cfg.get("msa_head_width", 16)
print(f"msa_encoder cfg: n_layers={n_layers} n_heads={n_heads} head_width={head_w}", flush=True)

sd = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    for k, v in load_file(f).items():
        if k.startswith("msa_encoder."):
            sd[k[len("msa_encoder."):]] = v
print(f"loaded {len(sd)} msa_encoder tensors", flush=True)

B, L, M = 1, 128, 256
d_pair = sd["blocks.0.outer_product_mean.Wout.weight"].shape[0]
d_inputs = sd["project_inputs.weight"].shape[1]
g = torch.Generator().manual_seed(0)
mk = lambda *s: torch.randn(*s, generator=g)
inputs = dict(
    x_pair=mk(B, L, L, d_pair),
    x_inputs=mk(B, L, d_inputs),
    msa_oh=torch.nn.functional.one_hot(torch.randint(0, 33, (B, L, M), generator=g), 33).float(),
    has_deletion=mk(B, L, M).gt(0).float(),
    deletion_value=mk(B, L, M).abs(),
    msa_mask=torch.ones(B, L, M),
)
print(f"d_pair={d_pair} d_inputs={d_inputs}  L={L} M={M}", flush=True)

def run(area):
    tenstorrent.SMALL_GRID_MSA_TILE_AREA = area
    mod = E.MSAEncoder(n_layers, n_heads, head_w)
    missing = mod.load_state_dict(sd, strict=False)
    rows = tenstorrent.msa_row_tile(L, M)
    print(f"  area={area} -> msa_row_tile={rows}", flush=True)
    out = mod(**inputs)
    return torch.Tensor(out).clone(), rows

single, r0 = run(0)
blocked, r1 = run(8192)
assert r0 == 0 and r1 == 32, f"arms did not differ: {r0} vs {r1}"
same = torch.equal(single, blocked)
diff = (single - blocked).abs().max().item()
print(f"RESULT: torch.equal={same}  max_abs_diff={diff}  shape={tuple(single.shape)}", flush=True)
print("PASS" if same else "FAIL", flush=True)

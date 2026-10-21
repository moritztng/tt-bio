"""Float64: the folded AE at N and at the 32-bucket width with masked pad keys gives the same valid rows."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fold import CKPT, load_fold  # noqa: E402

torch.set_default_dtype(torch.float64)
F = load_fold()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
for part in ("encoder", "decoder"):
    f = F.fold(sd, part)
    n_feat = f["in_w"].shape[1]
    g = torch.Generator().manual_seed(0)
    for N in (37, 250):
        Np = -(-N // 32) * 32
        feats = torch.randn(2, N, n_feat, generator=g)
        att = torch.ones(2, N, dtype=torch.bool)
        att[1, N - 5:] = False
        att[0, 2:4] = False
        x = F.heads_forward(f, F.trunk_forward(f, feats, att))
        fp = torch.zeros(2, Np, n_feat); fp[:, :N] = feats
        ap = torch.zeros(2, Np, dtype=torch.bool); ap[:, :N] = att
        xp = F.heads_forward(f, F.trunk_forward(f, fp, ap))[:, :N]
        d = (x - xp)[att].abs().max() / x[att].abs().max()
        print(part, N, Np, "max_rel", float(d))

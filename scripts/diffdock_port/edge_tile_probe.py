"""Localise a wrong edge-tile kernel: tap its Y, AB and Wt tiles and its output, and grade each
tile against the float64 schedule fed the same bf16-rounded inputs.

    TT_VISIBLE_DEVICES=0 ... timeout 600 python scripts/diffdock_port/edge_tile_probe.py \
        --layer rec_emb_layers.0 --edges 64 --out probe.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from device_tp_grade import CKPT, inputs, layer  # noqa: E402
from tt_bio.diffdock import config as dcfg  # noqa: E402
from tt_bio.diffdock import device as ddev  # noqa: E402
from tt_bio.diffdock.edge_tile import TILE, DeviceEdgeTile  # noqa: E402


def pcc(x, y):
    x, y = x.double().flatten(), y.double().flatten()
    if x.norm() == 0 and y.norm() == 0:
        return 1.0
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def expected(mod, ea, x1, x2):
    """float64 Y, AB, Wt and output tiles (kernel order) from bf16-rounded inputs, per group."""
    plan, T = mod.plan, TILE
    r = lambda t: t.to(torch.bfloat16).double()
    h = r(plan.hidden_act(r(ea)))
    a = torch.nn.functional.pad(r(x1), (0, plan.in_pad - x1.shape[1]))
    b = r(x2)[:, :plan.n_sh]
    out = []
    for g, s in zip(plan.groups, mod.sched):
        AG = a @ g.G
        Y = torch.zeros(a.shape[0], g.n_y * T, dtype=torch.float64)
        for t, (j, yi) in enumerate(zip(g.ag_j, g.ag_y)):
            Y[:, yi * T:(yi + 1) * T] += AG[:, t * T:(t + 1) * T] * b[:, j:j + 1]
        ab = s["ab"]
        AB = torch.cat([a[:, kt * T:(kt + 1) * T] * b[:, s["bj"][js]:s["bj"][js] + 1]
                        for js, kt in zip(ab[0::2], ab[1::2])], 1)
        Wt = h @ g.L2
        O = torch.zeros(a.shape[0], g.n_out_tiles * T, dtype=torch.float64)
        for wi, yi, ot, lane in g.mac:
            O[:, ot * T + lane] += (Wt[:, wi * T:(wi + 1) * T] * Y[:, yi * T:(yi + 1) * T]).sum(1)
        out.append({1: Y, 2: AB, 3: Wt, 0: O})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="rec_emb_layers.0")
    ap.add_argument("--edges", type=int, default=64)
    ap.add_argument("--taps", default="0,1,2,3")
    ap.add_argument("--out")
    args = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device

    cfg = dcfg.load(CKPT / "score_model", old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    gen = torch.Generator().manual_seed(0)
    E = ddev.pad32(args.edges)
    tp, fc = layer(cfg, sd, args.layer)
    ea, x1, x2 = inputs(tp, fc, E, gen)
    dev = get_device()
    ins = [ddev.to_device(dev, t, ttnn.bfloat16) for t in (ea, x1, x2)]
    res = {}
    for tap in (int(t) for t in args.taps.split(",")):
        mod = DeviceEdgeTile(dev, tp, fc, tap=tap)
        exp = expected(mod, ea, x1, x2)
        got = ttnn.to_torch(mod(*ins)).double()[:E]
        rows = []
        for gi, n in enumerate(mod.tap_n):
            ref = exp[gi][tap]
            for t in range(n):
                g_ = got[:, (mod.tap0[gi] + t) * TILE:(mod.tap0[gi] + t + 1) * TILE]
                r_ = ref[:, t * TILE:(t + 1) * TILE]
                rows.append(dict(group=gi, tile=t, pcc=round(pcc(g_, r_), 5),
                                 ref_norm=round(r_.norm().item(), 4),
                                 got_norm=round(g_.norm().item(), 4)))
        bad = [r for r in rows if r["pcc"] < 0.99]
        res[tap] = dict(n_tiles=len(rows), n_bad=len(bad), worst=sorted(rows, key=lambda r: r["pcc"])[:8],
                        all=rows)
        print("tap", tap, "tiles", len(rows), "bad", len(bad), json.dumps(res[tap]["worst"][:4]),
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

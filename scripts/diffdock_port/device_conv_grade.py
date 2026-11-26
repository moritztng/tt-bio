"""Grade one DiffDock-L conv layer on the card against the host module in float64.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:dfd-device \
        timeout 900 python scripts/diffdock_port/device_conv_grade.py --out conv.json

Real v1.1 weights (FC blocks, batch norm) for each named layer; a random graph whose edge
groups are tile-sized, a random 0/1 mask, one node with every edge masked (it must come out as
the residual alone, never NaN) and pad edges pointing at a pad node, as graph.py builds them.
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.diffdock import config as dcfg  # noqa: E402
from tt_bio.diffdock.conv import TensorProductConvLayer  # noqa: E402
from tt_bio.diffdock.harmonics import spherical_harmonics  # noqa: E402
from tt_bio.diffdock import device as ddev  # noqa: E402

CKPT = Path(os.environ.get("DIFFDOCK_MODELS", Path.home() / "scratch/dfd/ckpt/extracted"))


def load_layer(cfg, sd, name):
    p = next(x for x in cfg.layer_plan() if x["name"] == name)
    m = TensorProductConvLayer(p["in_irreps"], p.get("sh_irreps", str(cfg.sh_irreps)),
                               p["out_irreps"], p["n_edge_features"],
                               hidden_features=p["hidden_features"], edge_groups=p["edge_groups"],
                               residual=p.get("residual", True))
    pre = name + "."
    missing, _ = m.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
                                   strict=False)
    if missing:
        raise KeyError(f"{name}: checkpoint lacks {missing}")
    return m.eval().double()


def grade(ref, got):
    ref, got = ref.double().flatten(), got.double().flatten()
    return dict(pcc=torch.corrcoef(torch.stack([ref, got]))[0, 1].item(),
                max_rel_err=((got - ref).abs().max() / ref.abs().max()).item(),
                rel_frobenius=(torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item(),
                finite=bool(torch.isfinite(got).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="rec_emb_layers.0,conv_layers.0,conv_layers.10")
    ap.add_argument("--nodes", type=int, default=256)
    ap.add_argument("--group-edges", type=int, default=2048)
    ap.add_argument("--out")
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.diffdock.capture import clock_summary, sample_aiclk

    cfg = dcfg.load(CKPT / "score_model", old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    dev = get_device()
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_aiclk,
                          args=(stop, samples, os.environ.get("TT_VISIBLE_DEVICES", "0")),
                          daemon=True)
    th.start()
    gen = torch.Generator().manual_seed(1)
    N, Eg = args.nodes, args.group_edges
    pad_node, lonely = N - 1, 0
    rows = {}
    for name in args.layers.split(","):
        layer = load_layer(cfg, sd, name)
        G = layer.edge_groups
        E = G * Eg
        real = E - 64                                  # the last 64 edges are pad edges
        src = torch.randint(1, N - 1, (E,), generator=gen)
        dst = torch.randint(0, N - 1, (E,), generator=gen)
        src[:8] = lonely                               # node 0 has edges, all masked
        src[real:], dst[real:] = pad_node, pad_node
        mask = (torch.rand(E, generator=gen) < 0.7).double()
        mask[:8], mask[real:] = 0.0, 0.0
        ei = torch.stack([src, dst])
        x = torch.randn(N, layer.in_irreps.dim, generator=gen, dtype=torch.float64)
        x[pad_node] = 0
        sh = spherical_harmonics(torch.randn(E, 3, generator=gen, dtype=torch.float64), (0, 1, 2))
        nef = (layer.fc[0] if G > 1 else layer.fc)[0].weight.shape[1]
        groups = [torch.randn(Eg, nef, generator=gen, dtype=torch.float64) for _ in range(G)]
        with torch.no_grad():
            ref = layer(x, ei, groups, sh, edge_mask=mask)
        dl = ddev.DeviceConvLayer(dev, layer.float())
        layer.double()
        xp = torch.nn.functional.pad(x.float(), (0, ddev.pad32(x.shape[1]) - x.shape[1]))
        table = (ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev),
                 ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev))
        g = dl.graph(ei, N, N)
        got = dl(table, g, [ddev.to_device(dev, t) for t in groups], ddev.to_device(dev, sh),
                 ttnn.from_torch(mask.float().reshape(E, 1), dtype=ttnn.float32,
                                 layout=ttnn.TILE_LAYOUT, device=dev))
        got = ttnn.to_torch(got)[:, :layer.out_irreps.dim]
        real_rows = slice(0, N - 1)
        row = grade(ref[real_rows], got[real_rows])
        row["all_masked_node_is_residual"] = bool(torch.allclose(
            got[lonely].double(), ref[lonely], rtol=1e-2, atol=1e-2))
        row.update(edges=E, groups=G, nodes=N, mask_on=float(mask.mean()))
        rows[name] = row
        print(name, json.dumps(row), flush=True)
    stop.set()
    th.join(timeout=30)
    res = dict(rows=rows, chip=os.environ.get("TT_VISIBLE_DEVICES"), **clock_summary(samples))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

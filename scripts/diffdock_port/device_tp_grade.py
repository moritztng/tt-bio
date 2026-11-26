"""Grade DiffDock-L's edge tensor product on a Tenstorrent card against the host-torch module.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:dfd-device \
        timeout 900 python scripts/diffdock_port/device_tp_grade.py --out tp.json

For every distinct conv signature of the released score model, with that layer's real FC
weights from the v1.1 checkpoint: random node features, real spherical harmonics of random
edge vectors, random edge features. The golden is ``tp(x1, x2, fc(edge_attr))`` in float64
through :mod:`tt_bio.diffdock.tensor_product`, which is itself graded against e3nn. The
float32 host module is graded the same way and reported beside the card, so the card's error
reads against the host's own rounding.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.diffdock import config as dcfg  # noqa: E402
from tt_bio.diffdock.conv import TensorProductConvLayer  # noqa: E402
from tt_bio.diffdock.geometry import spherical_harmonics  # noqa: E402
from tt_bio.diffdock import device as ddev  # noqa: E402

CKPT = Path(os.environ.get("DIFFDOCK_MODELS", Path.home() / "scratch/dfd/ckpt/extracted"))
SIGNATURES = ["rec_emb_layers.0", "rec_emb_layers.1", "rec_emb_layers.2", "conv_layers.0",
              "conv_layers.10", "final_conv", "tor_bond_conv"]


def layer(cfg, sd, name):
    p = next(x for x in cfg.layer_plan() if x["name"] == name)
    m = TensorProductConvLayer(p["in_irreps"], p.get("sh_irreps", str(cfg.sh_irreps)),
                               p["out_irreps"], p["n_edge_features"],
                               hidden_features=p["hidden_features"],
                               edge_groups=p["edge_groups"])
    fc = m.fc[0] if p["edge_groups"] > 1 else m.fc
    pre = f"{name}.fc.0." if p["edge_groups"] > 1 else f"{name}.fc."
    fc.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)})
    return m.tp, fc.double()


def inputs(tp, fc, n_edges, gen):
    x1 = torch.randn(n_edges, tp.in_irreps.dim, generator=gen, dtype=torch.float64)
    vec = torch.randn(n_edges, 3, generator=gen, dtype=torch.float64)
    ls = [t.l for t in tp.sh_irreps]
    if [t.mul for t in tp.sh_irreps] == [1] * len(ls) and ls == list(range(len(ls))):
        x2 = spherical_harmonics(ls, vec)
    else:  # the torsion conv's sh is a full tensor product; any unit-scale block exercises it
        x2 = torch.randn(n_edges, tp.sh_irreps.dim, generator=gen, dtype=torch.float64)
    ea = torch.randn(n_edges, fc[0].weight.shape[1], generator=gen, dtype=torch.float64)
    return ea, x1, x2


def grade(ref, got):
    ref, got = ref.double().flatten(), got.double().flatten()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    return dict(pcc=pcc,
                max_rel_err=((got - ref).abs().max() / ref.abs().max()).item(),
                rel_frobenius=(torch.linalg.norm(got - ref) / torch.linalg.norm(ref)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=4096)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = dcfg.load(CKPT / "score_model", old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    gen = torch.Generator().manual_seed(0)
    E = ddev.pad32(args.edges)

    dev = None
    if not args.host_only:
        from tt_bio.main import ensure_p300_mesh_descriptor
        ensure_p300_mesh_descriptor()
        import ttnn
        from tt_bio.tenstorrent import get_device
        from tt_bio.diffdock.capture import clock_summary, sample_aiclk
        dev = get_device()
        dt = getattr(ttnn, args.dtype)
        samples, stop = [], threading.Event()
        visible = os.environ.get("TT_VISIBLE_DEVICES", "0")
        th = threading.Thread(target=sample_aiclk, args=(stop, samples, visible), daemon=True)
        th.start()

    rows = {}
    for name in SIGNATURES:
        tp, fc = layer(cfg, sd, name)
        ea, x1, x2 = inputs(tp, fc, E, gen)
        with torch.no_grad():
            ref = tp(x1, x2, fc(ea))                                 # float64 golden
            fc32 = fc.float()
            host32 = tp(x1.float(), x2.float(), fc32(ea.float()))
            fc.double()
            emu = ddev.host_emulation(tp, fc, ea, x1, x2)
        row = dict(signature=f"{tp.in_irreps} x {tp.sh_irreps} -> {tp.out_irreps}",
                   paths=len(tp.paths), weight_numel=tp.weight_numel, edges=E,
                   host_fp32=grade(ref, host32), rewrite_fp64=grade(ref, emu))
        if dev is not None:
            mod = ddev.DeviceEdgeTP(dev, tp, fc.float(), dtype=dt)
            ins = [ddev.to_device(dev, t, dt) for t in (ea, x1, x2)]
            out = mod(*ins)
            got = ttnn.to_torch(out)[:E, :tp.out_irreps.dim]
            row["card"] = grade(ref, got)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            reps = 5
            for _ in range(reps):
                mod(*ins)
            ttnn.synchronize_device(dev)
            row["card_ms_untraced"] = (time.perf_counter() - t0) / reps * 1e3
        rows[name] = row
        print(name, json.dumps(row), flush=True)

    res = dict(dtype=args.dtype, edges=E, rows=rows)
    if dev is not None:
        stop.set()
        th.join(timeout=30)
        res.update(clock_summary(samples))
        res["chip"] = os.environ.get("TT_VISIBLE_DEVICES")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

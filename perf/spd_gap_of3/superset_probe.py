"""OF3 atom transformer superset bias built on the chip against the host build it replaces.

For each atom count: random fp32 z_bias [1, nb, 4, 32, 128] x 3 and a 0 / -1e9 mask_bias, key
windows as OF3 places them (128 keys centred on each 32-atom block, shifted back inside the real
atoms at both ends). Runs `OF3AtomTransformer._superset` with on_device=True and False, prints
one JSON line per atom count: bit-equal, max abs diff, finite, and median ms of each build.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/spd_gap_of3/superset_probe.py [--atoms 1000,4001,8192]
       [--dtype bfloat16]
"""
import argparse, json, statistics, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--atoms", default="1000,4001,8192")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--dtype", default="float32", help="activation dtype: float32 (normal) or bfloat16 (--fast)")
ap.add_argument("--out", help="also append each JSON line here")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.openfold3_atom_transformer import OF3AtomTransformer

T.ADALN_S_HOIST = False              # the probe checks the bias only, not the AdaLN terms
dev = T.get_device()
at = object.__new__(OF3AtomTransformer)
at.device, at._act_dtype = dev, getattr(ttnn, A.dtype)
nq, nk, H = at.N_QUERY, at.N_KEY, at.N_HEADS


def up(t, layout=ttnn.TILE_LAYOUT, dt=ttnn.float32):
    return ttnn.from_torch(t, layout=layout, device=dev, dtype=dt)


def emit(rec):
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")


try:
    for n_atom in (int(x) for x in A.atoms.split(",")):
        g = torch.Generator().manual_seed(n_atom)
        nb = -(-n_atom // nq)
        ws = (torch.arange(nb) * nq + nq // 2 - nk // 2).clamp(0, max(n_atom - nk, 0))
        kidx = (ws[:, None] + torch.arange(nk)).reshape(1, nb * nk).to(torch.int32)
        kidx_tt = up(kidx, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        z_bias = [up(torch.randn(1, nb, H, nq, nk, generator=g) * 3) for _ in range(3)]
        valid = (ws[:, None] + torch.arange(nk)) < n_atom
        qok = (torch.arange(nb)[:, None] * nq + torch.arange(nq)) < n_atom
        mb = (1e9 * ((qok[:, :, None] & valid[:, None, :]).float() - 1)).reshape(1, nb, 1, nq, nk)
        mb_d = up(mb)
        res = {}
        for on_device in (False, True):
            ms, sup = [], None
            for _ in range(A.reps):
                if sup:
                    for bias, *_ in sup:
                        ttnn.deallocate(bias)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                sup = at._superset(z_bias, mb_d, kidx_tt, None, nb, on_device=on_device)
                ttnn.synchronize_device(dev)
                ms.append(1e3 * (time.perf_counter() - t0))
            res[on_device] = ([ttnn.to_torch(b).float() for b, *_ in sup], statistics.median(ms))
        host, dv = res[False][0], res[True][0]
        diff = max(float((a - b).abs().max()) for a, b in zip(host, dv))
        emit({"dtype": A.dtype, "n_atom": n_atom, "nb": nb, "W": int(host[0].shape[-1]),
              "bit_equal": all(torch.equal(a, b) for a, b in zip(host, dv)),
              "max_abs": diff, "finite": all(bool(torch.isfinite(b).all()) for b in dv),
              "host_ms": round(res[False][1], 2), "device_ms": round(res[True][1], 2)})
except Exception:
    traceback.print_exc()
    sys.exit(1)

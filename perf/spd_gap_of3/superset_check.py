"""OpenFold3 atom-transformer superset bias: `_superset_device` against `_superset_host`.

    python perf/spd_gap_of3/superset_check.py --out out/superset [--atoms 4000,4096,1000] [--reps 5]

Random pair bias and a random 0 / -1e9 mask in OpenFold3's block layout, the key windows placed the way OF3 places
them (128 keys centred on each 32-query block, shifted back inside the atoms at both ends). Reports bit-exactness of
the device bias against the host path (bf16 bits), max abs difference, and the median synced wall time of both.
"""
import argparse, json, math, statistics, time
from pathlib import Path

import torch


def windows(n_atom, nq=32, nk=128):
    nb = math.ceil(n_atom / nq)
    ws = (torch.arange(nb) * nq + nq // 2 - nk // 2).clamp(0, n_atom - nk)
    off = ws - torch.arange(nb) * nq
    lead = -(int(off.min()) // nq) * nq
    return nb, off + lead, lead + nk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", default="4000,4096,1000")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.openfold3_atom_transformer import OF3AtomTransformer
    dev = T.get_device()
    m = object.__new__(OF3AtomTransformer)
    m.device, m._act_dtype = dev, ttnn.bfloat16
    H = m.N_HEADS
    res, ok = {}, True
    for n_atom in (int(x) for x in a.atoms.split(",")):
        nb, off, W = windows(n_atom)
        g = torch.Generator().manual_seed(n_atom)
        zb = ttnn.from_torch(torch.randn(1, nb, H, 32, 128, generator=g) * 3, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        mb = ttnn.from_torch(1e9 * ((torch.rand(1, nb, 1, 32, 128, generator=g) > 0.2).float() - 1),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ts = {}
        for name, fn in (("host", m._superset_host), ("device", m._superset_device)):
            t = []
            for _ in range(a.reps):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                y = fn(zb, mb, off, W, nb)
                ttnn.synchronize_device(dev)
                t.append(time.perf_counter() - t0)
            ts[name] = (statistics.median(t), ttnn.to_torch(y))
        h, d = ts["host"][1], ts["device"][1]
        finite = bool(torch.isfinite(d.float()).all())
        same = h.shape == d.shape and torch.equal(h.view(torch.int16), d.view(torch.int16))
        diff = (h.float() - d.float()).abs()
        r = {"nb": nb, "W": W, "shape": list(d.shape), "runs": len(set(off.tolist())), "finite": finite,
             "bit_exact": same, "n_diff": int((diff > 0).sum()), "max_abs": float(diff.max()),
             "host_s": ts["host"][0], "device_s": ts["device"][0]}
        ok &= finite and r["max_abs"] <= 1e-2 * float(h.float().abs().max())
        res[n_atom] = r
        print(n_atom, r, flush=True)
    res["verdict"] = "PASS" if ok else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(res["verdict"])


if __name__ == "__main__":
    main()

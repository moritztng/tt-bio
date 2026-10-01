#!/usr/bin/env python3
"""Bytes by site and dtype for one Evoformer block, as the BindCraft 2 round runs it.

Every outermost ttnn / ttnn.experimental call is charged the DRAM bytes of its device-tensor
operands and results at padded tile shape and stored dtype (one pass each: a floor for ops that
re-read). Keyed by (phase, op, issuing tt_bio site, dtypes in, dtype out). The block runs
checkpointed under `bindcraft2.fast_round()`, warm (the first pass is discarded): "fwd" is the
untaped forward, "bwd" the backward with its taped recompute. This is what a dtype lever can
reach, site by site; it says nothing about time.
"""
import argparse, collections, json, pathlib, sys, traceback
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SKIP = {"from_torch", "to_torch", "to_device", "from_device", "synchronize_device", "deallocate",
        "allocate_tensor_on_device", "reallocate", "get_device_tensors", "reshape", "view"}
SZ = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1,
      "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}


def site():
    fr = [f for f in traceback.extract_stack()[:-3] if "/tt_bio/" in f.filename]
    if not fr:
        return "?"
    f = fr[-1]
    s = f"{f.filename.split('/tt_bio/')[-1]}:{f.lineno}:{f.name}"
    if len(fr) > 1:
        u = fr[-2]
        s += f" <- {u.filename.split('/tt_bio/')[-1]}:{u.lineno}:{u.name}"
    return s


def tbytes(t, ttnn):
    if not isinstance(t, ttnn.Tensor) or t.storage_type() != ttnn.StorageType.DEVICE:
        return None
    try:
        if t.memory_config().buffer_type != ttnn.BufferType.DRAM:
            return None
    except Exception:
        return None
    sh = list(t.padded_shape)
    n = 1
    for d in sh:
        n *= d
    dt = str(t.dtype).split(".")[-1]
    return dt, n * SZ.get(dt, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from ttnn.decorators import FastOperation
    from tt_bio import bindcraft2
    from tt_bio.af2 import af2_pair_masks
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    n, nreal = a.n, a.n - 27
    torch.manual_seed(0)
    m0, z0 = torch.randn(a.depth, n, 256) * 0.5, torch.randn(n, n, 128) * 0.5
    seq = torch.zeros(n); seq[:nreal] = 1
    msa_mask = tr.up(seq.expand(a.depth, n).contiguous())
    pm = af2_pair_masks(seq[:, None] * seq[None, :], tr.device)
    blk = tr.model.device_evoformer[0]
    st = {"phase": "off", "depth": 0}
    rows = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])

    def wrap(name, fn):
        def w(*args, **kw):
            if st["phase"] == "off" or st["depth"]:
                return fn(*args, **kw)
            st["depth"] += 1
            try:
                out = fn(*args, **kw)
            finally:
                st["depth"] -= 1
            flat = []
            for x in list(args) + list(kw.values()):
                flat.extend(x if isinstance(x, (list, tuple)) else [x])
            ins = [x for x in (tbytes(t, ttnn) for t in flat) if x]
            outs = out if isinstance(out, (list, tuple)) else [out]
            outs = [x for x in (tbytes(o, ttnn) for o in outs) if x]
            key = (st["phase"], name, site(), ",".join(sorted({d for d, _ in ins})),
                   ",".join(sorted({d for d, _ in outs})))
            r = rows[key]
            r[0] += 1
            for d, b in ins:
                r[2]["r:" + d] += b
            for d, b in outs:
                r[2]["w:" + d] += b
            r[1] += sum(b for _, b in ins + outs)
            return out
        return w

    for mod in (ttnn, ttnn.experimental):
        pre = "" if mod is ttnn else "experimental."
        for v in dir(mod):
            f = getattr(mod, v, None)
            if isinstance(f, FastOperation) and v not in SKIP:
                setattr(mod, v, wrap(pre + v, f))
    with bindcraft2.fast_round() as armed:
        for rep in range(2):
            st["phase"] = "fwd" if rep else "off"
            ml, zl = tr.leaf(m0), tr.leaf(z0)
            with T.tape():
                m, z = ag.checkpoint(lambda x, y: blk(x, y, msa_mask, *pm), ml, zl)
            tr.sync()
            seeds = [tr.seed(torch.randn(m0.shape) * 1e-3, m), tr.seed(torch.randn(z0.shape) * 1e-3, z)]
            st["phase"] = "bwd" if rep else "off"
            ag.backward([m, z], seeds)
            tr.sync()
            st["phase"] = "off"
            ag.release_pins()
    tot = collections.Counter()
    for (ph, *_), (c, b, by) in rows.items():
        tot[ph] += b
        for k, v in by.items():
            tot[f"{ph}:{k}"] += v
    out = {"n": n, "depth": a.depth, "armed": {k: str(v) for k, v in armed.items()},
           "totals_gb": {k: v / 1e9 for k, v in sorted(tot.items())},
           "rows": [{"phase": k[0], "op": k[1], "site": k[2], "in": k[3], "out": k[4], "calls": c,
                     "gb": b / 1e9, "by": {kk: vv / 1e9 for kk, vv in by.items()}}
                    for k, (c, b, by) in sorted(rows.items(), key=lambda kv: -kv[1][1])]}
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["totals_gb"].items():
        print(f"{k:24s} {v:9.4f} GB")


if __name__ == "__main__":
    main()

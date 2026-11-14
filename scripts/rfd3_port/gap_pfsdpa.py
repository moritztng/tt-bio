#!/usr/bin/env python3
"""gap_pfsdpa: the token encoder's pairformer attention, manual fp32 chain against the fused SDPA (RFD3_PF_SDPA),
each with the pair bias from the linear and from dit_pair_bias (RFD3_PF_DPB). Random weights at the checkpoint's
shapes, every input bf16-representable; error against a float64 reference of the same inputs. Sizes: the 685-token
page and 418 tokens (partial last tile).
    python scripts/rfd3_port/gap_pfsdpa.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_pfsdpa.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CS, CZ, H = 384, 128, 16
HD = CS // H


def weights(g):
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).bfloat16().float()  # noqa: E731
    w = {f"to_{k}.weight": r(CS, CS, sc=CS ** -0.5) for k in ("q", "k", "v", "a")}
    w["to_g.0.weight"] = r(CS, CS, sc=CS ** -0.5)
    w["to_q.bias"] = r(CS, sc=0.1)
    w["to_b.weight"] = r(H, CZ, sc=CZ ** -0.5)
    for k in ("ln_1.weight", "to_k.ln.weight", "to_v.ln.weight"):
        w[k] = (1 + r(CS, sc=0.1)).bfloat16().float()
    w["ln_0.weight"] = (1 + r(CZ, sc=0.1)).bfloat16().float()
    return w


def reference(w, s, z):
    d = {k: v.double() for k, v in w.items()}
    rms = lambda x, g: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g  # noqa: E731
    a = rms(s.double(), d["ln_1.weight"])
    q = a @ d["to_q.weight"].t() + d["to_q.bias"]
    k = rms(a @ d["to_k.weight"].t(), d["to_k.ln.weight"])
    v = rms(a @ d["to_v.weight"].t(), d["to_v.ln.weight"])
    b = rms(z.double(), d["ln_0.weight"]) @ d["to_b.weight"].t()          # [1, I, I, H]
    heads = lambda x: x.reshape(1, -1, H, HD).transpose(1, 2)              # noqa: E731
    sc = heads(q) @ heads(k).transpose(-1, -2) / HD ** 0.5 + b.permute(0, 3, 1, 2)
    o = (sc.softmax(-1) @ heads(v)).transpose(1, 2).reshape(1, -1, CS)
    o = o * torch.sigmoid(a @ d["to_g.0.weight"].t())
    return o @ d["to_a.weight"].t()


def case(dev, ckc, I):
    g = torch.Generator().manual_seed(I)
    w = weights(g)
    s_t = torch.randn(1, I, CS, generator=g).bfloat16()
    z_t = torch.randn(1, I, I, CZ, generator=g).bfloat16()
    ref = reference(w, s_t, z_t)
    attn = M.PairformerAttention(w, ckc, dtype=ttnn.bfloat16)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    s, z = up(s_t), up(z_t)
    rec, outs = {"I": I}, {}
    for sdpa in (False, True):
        for dpb in (False, True):
            name = f"{'sdpa' if sdpa else 'manual'}-{'dpb' if dpb else 'lin'}"
            M._PF_SDPA, M._PF_DPB = sdpa, dpb
            try:
                o = attn(s, z)
                ttnn.synchronize_device(dev)
                ts = []
                for _ in range(N):
                    ttnn.deallocate(o)
                    t0 = time.perf_counter()
                    o = attn(s, z)
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) * 1e3)
                d = ttnn.to_torch(o).double()[:, :I, :CS]
                ttnn.deallocate(o)
                e = d - ref
                r = {"ms_median": statistics.median(ts), "ms_min": min(ts),
                     "rel_err_f64": float(e.norm() / ref.norm()), "max_abs_f64": float(e.abs().max()),
                     "finite": bool(torch.isfinite(d).all())}
                outs[name] = d
            except Exception as ex:                                      # noqa: BLE001
                r = {"error": repr(ex)[:400]}
            rec[name] = r
            print(I, name, r, flush=True)
    return rec


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    rec = {"reps": N, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
           "arch": str(dev.arch()), "cases": [case(dev, ckc, I) for I in (685, 418)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""of3t-denoise: which op of the taped diffusion conditioning FORWARD returns garbage?

    probe_dcfwd.py --out F.json [--tokens N] [--reps R]

The DN384F / DN384FB A/A disagreed, and a two-process dump (probe_aa.py) put the difference in the
conditioning's si/zij: face-granular blocks off by up to 84 in a tensor of typical magnitude 19,
never on pads, different blocks each run, while the untaped rollout's conditioning is
bit-identical across processes. This runs `pair` and `single` stage by stage, as the shipped
methods do, untaped and on the tape, R times each in one process on the same seeded inputs, and
scores every stage against a float64 forward of the SAME device inputs read back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    n = int(argv[argv.index("--tokens") + 1]) if "--tokens" in argv else 64
    reps = int(argv[argv.index("--reps") + 1]) if "--reps" in argv else 2
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device, device_dtype_override
    import tt_bio.openfold3_diffusion as odm
    from tt_bio.openfold3_weights import _sub
    import tt_bio.openfold3_sample_diffusion as sdm

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = torch.load(Path.home() / "of3-weights/of3-p2-155k.pt", map_location="cpu",
                    weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    pre = "diffusion_module.diffusion_conditioning"
    with device_dtype_override(ttnn.float32):
        dc = odm.OF3DiffusionConditioning(_sub(sd, pre), ckc)
    W = {k[len(pre) + 1:]: v.double() for k, v in sd.items() if k.startswith(pre + ".")}
    g = torch.Generator().manual_seed(0)
    bf = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa
    s_trunk = torch.randn(1, n, 384, generator=g)
    s_input = torch.randn(1, n, 449, generator=g)
    n_emb = torch.randn(1, 1, 256, generator=g)
    z_trunk = torch.randn(1, n, n, 128, generator=g) * 19
    relpos = (torch.rand(1, n, n, 139, generator=g) > 0.9).float()
    tok = torch.ones(1, n, 1)
    pm = tok.reshape(1, n, 1, 1) * tok.reshape(1, 1, n, 1)
    raw = lambda t: t.value if isinstance(t, ag.Tensor) else t  # noqa: E731
    host = lambda t: ttnn.to_torch(raw(t)).double()  # noqa: E731

    def ln(x, w, b=None):
        mu = x.mean(-1, keepdim=True)
        v = ((x - mu) ** 2).mean(-1, keepdim=True)
        y = (x - mu) / torch.sqrt(v + 1e-5) * w
        return y if b is None else y + b

    def run(taped):
        T = odm.ttnn  # the shim inside a tape, ttnn outside
        wrap = (lambda t: ag.Tensor(t)) if taped else (lambda t: t)
        c = lambda x: sdm.ttnn.typecast(x, ttnn.float32) if raw(x).dtype == ttnn.bfloat16 else x  # noqa
        st = {}
        zt, rp = c(wrap(bf(z_trunk))), c(wrap(bf(relpos)))
        pmd = c(bf(pm))
        st["in_z"], st["in_relpos"] = host(zt), host(rp)
        zc = T.concat([zt, rp], dim=-1); st["z.concat"] = host(zc)
        z = T.layer_norm(zc, weight=dc.ln_z, epsilon=1e-5, compute_kernel_config=dc.compute_kernel_config)
        st["z.ln"] = host(z)
        z = dc._lin(z, dc.w_lin_z); st["z.lin"] = host(z)
        for i, tr in enumerate(dc.tr_z):
            x = T.layer_norm(z, weight=tr.ln_w, bias=tr.ln_b, epsilon=1e-5,
                             compute_kernel_config=tr.compute_kernel_config)
            st[f"z.tr{i}.ln"] = host(x)
            a = tr._lin(x, tr.la, activation="silu"); st[f"z.tr{i}.a"] = host(a)
            b = tr._lin(x, tr.lb); st[f"z.tr{i}.b"] = host(b)
            h = T.multiply(a, b); st[f"z.tr{i}.h"] = host(h)
            o = tr._lin(h, tr.lo); st[f"z.tr{i}.o"] = host(o)
            o = T.multiply(o, pmd); st[f"z.tr{i}.mask"] = host(o)
            z = T.add(z, o); st[f"z.tr{i}.add"] = host(z)
        # the shipped method end to end, for the deallocate pattern the stage copy lacks
        zs = dc.pair(c(wrap(bf(z_trunk))), c(wrap(bf(relpos))), pmd)
        st["z.shipped"] = host(zs)
        ss = dc.single(c(wrap(bf(s_trunk))), c(wrap(bf(s_input))), c(wrap(bf(n_emb))), c(bf(tok)))
        st["s.shipped"] = host(ss)
        return st

    runs = []
    for r in range(reps):
        runs.append(("untaped", run(False)))
        with ag.tape():
            runs.append(("taped", run(True)))

    # float64 of the SAME device inputs
    base = runs[0][1]
    f = {}
    zc = torch.cat([base["in_z"], base["in_relpos"]], -1)
    z = ln(zc, W["layer_norm_z.weight"]) @ W["linear_z.weight"].T
    f["z.lin"] = z
    for i in range(2):
        p = f"transition_z.{i}."
        x = ln(z, W[p + "layer_norm.weight"], W[p + "layer_norm.bias"])
        a = torch.nn.functional.silu(x @ W[p + "swiglu.linear_a.weight"].T)
        b = x @ W[p + "swiglu.linear_b.weight"].T
        z = z + ((a * b) @ W[p + "linear_out.weight"].T) * pm.double()
        f[f"z.tr{i}.add"] = z
    f["z.shipped"] = z

    rec = {"tokens": n, "reps": reps, "runs": []}
    ref0 = runs[0][1]
    for kind, st in runs:
        row = {"kind": kind, "stages": {}}
        for k, v in st.items():
            d = (v - ref0[k]).abs()
            e = {"eq_run0": bool(torch.equal(v, ref0[k])), "max_vs_run0": float(d.max()),
                 "n_vs_run0": int((d > 0).sum())}
            if k in f:
                e["rel_f64"] = float((v - f[k]).norm() / f[k].norm())
                e["max_f64"] = float((v - f[k]).abs().max())
            row["stages"][k] = e
        rec["runs"].append(row)
        print(kind, json.dumps({k: (v["eq_run0"], round(v["max_vs_run0"], 4), v.get("rel_f64"))
                                for k, v in row["stages"].items()}), flush=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

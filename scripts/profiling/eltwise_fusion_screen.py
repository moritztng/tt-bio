"""Op-level screen v2 for the three ttnn fusions available at the 0.68.0 pin.

v2 fixes two v1 defects:
  * ``addcmul`` takes ``value`` as a KEYWORD arg; v1 passed it positionally and every
    shape came back REFUSED on a signature mismatch, not a capability limit.
  * v1 screened addalpha at scale 64**-0.5 = 0.125, a power of two, which bf16 multiplies
    EXACTLY -- so it read bit-exact for a reason no real call site shares. Every actual
    site's head_dim is 48 or 32 (scale 0.1443.. / 0.1768..), screened here.

Per fusion/shape/scale: does the op accept the site's broadcast form, max-abs-diff of fused
vs the two-op chain, which arm is closer to a float64 reference, and per-call time with the
arms INTERLEAVED per rep (memory op-ab-must-interleave-arms-compile-warmup-bias).
"""
import os, sys, time, json
import torch
import ttnn
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor(None, int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
from tt_bio.tenstorrent import get_device

DEV = get_device()
CK = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                      math_approx_mode=False, fp32_dest_acc_en=True,
                                      packer_l1_acc=False)
REPS = int(os.environ.get("SCREEN_REPS", "12"))
rows = []


def _dev(t, dt):
    return ttnn.from_torch(t.to(torch.float32), dtype=dt, layout=ttnn.TILE_LAYOUT,
                           device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _one(fn):
    ttnn.synchronize_device(DEV)
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(DEV)
    dt = time.perf_counter() - t0
    ttnn.deallocate(out)
    return dt


def screen(name, site, dtype, devs, chain, fused, ref64):
    try:
        f_out = fused(devs)
    except Exception as e:
        rows.append(dict(fusion=name, site=site, dtype=str(dtype).split(".")[-1],
                         status="REFUSED", err=str(e)[:300]))
        print(f"  REFUSED {name} {site}: {str(e)[:200]}", flush=True)
        return
    c_out = chain(devs)
    f = ttnn.to_torch(f_out).to(torch.float64)
    c = ttnn.to_torch(c_out).to(torch.float64)
    r = ref64
    assert f.shape == r.shape, f"fused shape {f.shape} != ref {r.shape}"
    assert c.shape == r.shape, f"chain shape {c.shape} != ref {r.shape}"
    d_fc = float((f - c).abs().max())
    e_f = float((f - r).abs().max())
    e_c = float((c - r).abs().max())
    scale = float(r.abs().max()) or 1.0
    ttnn.deallocate(f_out); ttnn.deallocate(c_out)
    tf = tc = 0.0
    for i in range(REPS):
        if i % 2 == 0:
            tf += _one(lambda: fused(devs)); tc += _one(lambda: chain(devs))
        else:
            tc += _one(lambda: chain(devs)); tf += _one(lambda: fused(devs))
    tf /= REPS; tc /= REPS
    row = dict(fusion=name, site=site, dtype=str(dtype).split(".")[-1], status="OK",
               bit_exact=(d_fc == 0.0), diff_fused_vs_chain=d_fc, rel_diff=d_fc / scale,
               err_fused_vs_f64=e_f, err_chain_vs_f64=e_c,
               closer_to_f64=("fused" if e_f < e_c else "chain" if e_c < e_f else "tie"),
               fused_us=tf * 1e6, chain_us=tc * 1e6, saved_us=(tc - tf) * 1e6,
               ratio=tc / tf if tf else 0.0)
    rows.append(row)
    print(f"  {name:11s} {site:52s} {row['dtype']:9s} diff={d_fc:.3e} (rel {row['rel_diff']:.2e}) "
          f"f64err fused={e_f:.3e} chain={e_c:.3e} closer={row['closer_to_f64']:5s} | "
          f"{tf*1e6:8.1f} vs {tc*1e6:8.1f} us -> saves {(tc-tf)*1e6:7.1f} us ({row['ratio']:.3f}x)",
          flush=True)


# ------------------------------------------------------------------ addalpha (TST)
# site: sc = multiply(sc, scale); sc = add(sc, bias)  ->  addalpha(bias, sc, scale)
def run_tst(site, sc_shape, b_shape, sc_scale, dtype):
    torch.manual_seed(0)
    sc64 = torch.randn(*sc_shape, dtype=torch.float64) * 8.0
    b64 = torch.randn(*b_shape, dtype=torch.float64) * 2.0
    d = {"sc": _dev(sc64, dtype), "b": _dev(b64, dtype)}
    # float64 reference taken from the bf16-ROUNDED inputs, so the only difference the
    # screen reports is the arithmetic, not the input quantisation.
    sc_q = ttnn.to_torch(d["sc"]).to(torch.float64)
    b_q = ttnn.to_torch(d["b"]).to(torch.float64)
    ref = sc_q * sc_scale + b_q
    screen("addalpha", site, dtype, d,
           lambda x: ttnn.add(ttnn.multiply(x["sc"], sc_scale), x["b"]),
           lambda x: ttnn.addalpha(x["b"], x["sc"], sc_scale), ref)
    for v in d.values():
        ttnn.deallocate(v)


# ------------------------------------------------------------------ addcmul (TTT)
# site: out = multiply(out, mask); x = add(x, out)  ->  addcmul(x, out, mask, value=1.0)
def run_ttt(site, x_shape, m_shape, dtype):
    torch.manual_seed(0)
    x64 = torch.randn(*x_shape, dtype=torch.float64)
    o64 = torch.randn(*x_shape, dtype=torch.float64)
    m64 = (torch.rand(*m_shape, dtype=torch.float64) > 0.2).to(torch.float64)
    d = {"x": _dev(x64, dtype), "o": _dev(o64, dtype), "m": _dev(m64, dtype)}
    xq = ttnn.to_torch(d["x"]).to(torch.float64)
    oq = ttnn.to_torch(d["o"]).to(torch.float64)
    mq = ttnn.to_torch(d["m"]).to(torch.float64)
    ref = xq + oq * mq
    screen("addcmul", site, dtype, d,
           lambda z: ttnn.add(z["x"], ttnn.multiply(z["o"], z["m"])),
           lambda z: ttnn.addcmul(z["x"], z["o"], z["m"], value=1.0), ref)
    for v in d.values():
        ttnn.deallocate(v)


# ------------------------------------------------------------- layer_norm residual
_LNW = {}
def _lnw(c, dt):
    if (c, dt) not in _LNW:
        _LNW[(c, dt)] = (_dev(torch.ones(c, dtype=torch.float64), dt),
                         _dev(torch.zeros(c, dtype=torch.float64), dt))
    return _LNW[(c, dt)]

def run_ln(site, shape, dtype):
    torch.manual_seed(0)
    d = {"a": _dev(torch.randn(*shape, dtype=torch.float64), dtype),
         "b": _dev(torch.randn(*shape, dtype=torch.float64), dtype)}
    w, b = _lnw(shape[-1], dtype)
    aq = ttnn.to_torch(d["a"]).to(torch.float64)
    bq = ttnn.to_torch(d["b"]).to(torch.float64)
    x = aq + bq
    ref = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
    screen("ln_residual", site, dtype, d,
           lambda z: ttnn.layer_norm(ttnn.add(z["a"], z["b"]), weight=w, bias=b,
                                     epsilon=1e-5, compute_kernel_config=CK),
           lambda z: ttnn.layer_norm(z["a"], residual_input_tensor=z["b"], weight=w, bias=b,
                                     epsilon=1e-5, compute_kernel_config=CK), ref)
    for v in d.values():
        ttnn.deallocate(v)


if __name__ == "__main__":
    bf16, f32 = ttnn.bfloat16, ttnn.float32
    S48, S32, S64 = 48 ** -0.5, 32 ** -0.5, 64 ** -0.5
    print("== addalpha (TST): multiply-by-scalar + add -> one fused binary ==", flush=True)
    # of3 diffusion transformer / protenix DiT: head_dim 48, scale NOT a power of two
    run_tst("of3_dit  sc[1,16,512,512] b same    hd48", (1, 16, 512, 512), (1, 16, 512, 512), S48, bf16)
    run_tst("of3_dit  sc[1,16,512,512] b hd-bcst hd48", (1, 16, 512, 512), (1, 1, 512, 512), S48, bf16)
    run_tst("of3_dit  sc[1,16,256,256] b same    hd48", (1, 16, 256, 256), (1, 16, 256, 256), S48, bf16)
    # protenix / of3 atom transformer: head_dim 32
    run_tst("prot_att sc[1,4,512,512]  b same    hd32", (1, 4, 512, 512), (1, 4, 512, 512), S32, bf16)
    run_tst("of3_atom sc[1,64,4,32,128] b bcst   hd32", (1, 64, 4, 32, 128), (1, 1, 4, 32, 128), S32, bf16)
    # protenix DiT fp32 raw-matmul path (tenstorrent.py:7849)
    run_tst("prot_dit_fp32 sc[1,16,512,512] hd48", (1, 16, 512, 512), (1, 16, 512, 512), S48, f32)
    # the v1 power-of-two control, kept so the artifact is on the record
    run_tst("CONTROL pow2 scale 0.125 (no real site)", (1, 16, 512, 512), (1, 16, 512, 512), S64, bf16)

    print("== addcmul (TTT): multiply-by-mask + add-residual -> one fused ternary ==", flush=True)
    run_ttt("of3_dit   x[1,512,768]  mask col [1,512,1]", (1, 512, 768), (1, 512, 1), bf16)
    run_ttt("of3_dit   x[1,512,768]  mask full", (1, 512, 768), (1, 512, 768), bf16)
    run_ttt("of3_atom  x[1,4096,128] mask col [1,4096,1]", (1, 4096, 128), (1, 4096, 1), bf16)
    run_ttt("of3_dmod  x[1,64,32,128,16] mask col", (1, 64, 32, 128, 16), (1, 64, 32, 128, 1), bf16)

    print("== layer_norm(residual_input_tensor=) : add + norm -> one kernel ==", flush=True)
    run_ln("protenix_v1 conf head [1,512,512,128]", (1, 512, 512, 128), bf16)
    run_ln("protenix_v2 conf head [1,256,256,256]", (1, 256, 256, 256), bf16)

    out = os.environ.get("SCREEN_OUT", "/tmp/eltwise_screen.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {out}", flush=True)

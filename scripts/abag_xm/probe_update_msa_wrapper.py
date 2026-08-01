"""Reproduce update_msa's two branches directly, with the real modules and synthetic weights.

Runs in seconds instead of loading a checkpoint and folding, which is why the whole chunking question
was eventually settled here rather than by more 4-minute folds.

What it established, in order:
  * PWA alone, including the single reused ttnn.clone(z) shared across chunks -- BIT-EXACT.
  * Transition alone -- bit-exact at some shapes, not at others (see probe_transition_width.py).
  * PWA and Transition composed per chunk, exactly as update_msa runs them -- differs only where
    Transition does.

And the conclusion that supersedes all of it: the difference is **one bf16 mantissa step, not a
defect** -- probe_transition_vs_torch.py shows both paths equally close to an fp32 reference.

The positive control in probe_full_update_msa is load-bearing and should not be removed. It asserts
the comparator can see a known-different input BEFORE any verdict is reported. It fired on its first
run -- perturbing a bf16 element of magnitude ~1000 by +1.0 rounds back to the same value, so the
"known-different" input was not different -- and an earlier missing control let a broken comparison
report vacuous "identical" for two whole passes.

Run on the Galaxy with one chip free:
    TT_VISIBLE_DEVICES=<n> python3 -u scripts/abag_xm/probe_update_msa_wrapper.py
"""
import os
import torch
import ttnn

from tt_bio.tenstorrent import (get_device, MSA_CHUNK_SIZE, PairWeightedAveraging,
                                Transition, _dtype)

D = int(os.environ.get("PROBE_DEPTH", 8998))
S = int(os.environ.get("PROBE_TOKENS", 288))
C_M = 128
C_Z = 384
HD, NH = 8, 8
CHUNK = int(os.environ.get("PROBE_CHUNK", MSA_CHUNK_SIZE))


def report(name, a, b):
    if torch.equal(a, b):
        print(f"  {name:38s} BIT-EXACT")
        return True
    d = (a.float() - b.float()).abs()
    print(f"  {name:38s} DIFFERS  maxabs {d.max():.3e}  frac {(d > 0).float().mean():.4f}")
    return False


def main():
    dev = get_device()
    torch.manual_seed(0)
    print(f"m=(1,{D},{S},{C_M})  z=(1,{S},{S},{C_Z})  chunk={CHUNK}  head_dim={HD} n_heads={NH}")

    # torch_to_tt transposes by default, so store each weight already transposed relative to use.
    w = {
        "norm_m.weight": torch.randn(C_M), "norm_m.bias": torch.randn(C_M),
        "norm_z.weight": torch.randn(C_Z), "norm_z.bias": torch.randn(C_Z),
        "proj_m.weight": torch.randn(NH * HD, C_M),
        "proj_g.weight": torch.randn(NH * HD, C_M),
        "proj_z.weight": torch.randn(NH, C_Z),
        "proj_o.weight": torch.randn(C_M, NH * HD),
    }
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)
    try:
        pwa = PairWeightedAveraging(HD, NH, w, kc)
    except Exception as e:                                   # noqa: BLE001
        print("could not build PairWeightedAveraging from a plain dict:", repr(e)[:200])
        return

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dtype())

    m_host = torch.randn(1, D, S, C_M)
    z_host = torch.randn(1, S, S, C_Z)

    # --- whole branch, exactly as update_msa writes it
    m = up(m_host)
    z = up(z_host)
    whole = ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
    whole_t = ttnn.to_torch(whole)
    ttnn.deallocate(whole)
    ttnn.deallocate(m)

    # --- chunked branch, exactly as update_msa writes it (ONE clone reused across chunks)
    m = up(m_host)
    zc = ttnn.clone(z)
    parts = []
    for s in range(0, D, CHUNK):
        mc = m[:, s:min(s + CHUNK, D), :, :]
        mc = ttnn.add(mc, ttnn.reshape(pwa(mc, zc), tuple(mc.shape)))
        parts.append(mc)
    chunked_t = ttnn.to_torch(ttnn.concat(parts, dim=1))

    if torch.equal(whole_t, chunked_t):
        print("  wrapper(PWA only, reused clone)   BIT-EXACT"
              "  -> the wrapper reproduces the whole path exactly")
    else:
        d = (whole_t.float() - chunked_t.float()).abs()
        print(f"  wrapper(PWA only, reused clone)   DIFFERS  maxabs {d.max():.3e} "
              f" frac {(d > 0).float().mean():.4f}  -> the wrapper itself differs")

        # Narrow it: clone z per call, the way the whole branch does.
        m2 = up(m_host)
        parts2 = []
        for s in range(0, D, CHUNK):
            mc = m2[:, s:min(s + CHUNK, D), :, :]
            mc = ttnn.add(mc, ttnn.reshape(pwa(mc, ttnn.clone(z)), tuple(mc.shape)))
            parts2.append(mc)
        c2 = ttnn.to_torch(ttnn.concat(parts2, dim=1))
        print("  same, but cloning z PER CHUNK    ",
              "BIT-EXACT -> the REUSED clone is the bug"
              if torch.equal(whole_t, c2) else "still DIFFERS -> not the clone")

    probe_transition()
    probe_full_update_msa()
    print("done")




def probe_transition():
    """Transition alone under depth chunking.

    Transition splits its own H axis internally (ttnn.chunk into ceil(H/transition_h_chunk_size)
    parts, then concat), so handing it H=512 instead of H=8998 changes that internal split. Whether
    it differs depends on the (depth, width) pair -- see probe_transition_width.py for the matrix.

    A DIFFERS here is NOT a defect: probe_transition_vs_torch.py measured both paths equally close
    to an fp32 reference, one bf16 mantissa step apart.
    """
    dev = get_device()
    torch.manual_seed(0)
    H = 4 * C_M
    w = {"norm.weight": torch.randn(C_M), "norm.bias": torch.randn(C_M),
         "fc1.weight": torch.randn(H, C_M), "fc2.weight": torch.randn(H, C_M),
         "fc3.weight": torch.randn(C_M, H)}
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)
    tr = Transition(w, kc)
    print(f"\ntransition probe: x=(1,{D},{S},{C_M})  hidden={H}  chunk={CHUNK}")
    x_host = torch.randn(1, D, S, C_M)
    x = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dtype())
    whole_t = ttnn.to_torch(tr(x))
    parts = [tr(x[:, s:min(s + CHUNK, D), :, :]) for s in range(0, D, CHUNK)]
    chunked_t = ttnn.to_torch(ttnn.concat(parts, dim=1))
    if torch.equal(whole_t, chunked_t):
        print("  Transition(depth-chunked)         BIT-EXACT")
    else:
        d = (whole_t.float() - chunked_t.float()).abs()
        print(f"  Transition(depth-chunked)         DIFFERS  maxabs {d.max():.3e} "
              f" frac {(d > 0).float().mean():.4f}  (one bf16 step; not a defect)")




def probe_full_update_msa():
    """The untested unit: PWA **and** Transition together, per chunk, exactly as update_msa runs.

    The real fold's first divergence is `block0:m_feat`, the direct output of update_msa in cycle 0's
    first MSA block, so this reproduces both branches verbatim rather than testing a component.
    It tracks whatever Transition does: identical where Transition is identical.

    The positive control below is load-bearing -- see the module docstring.
    """
    dev = get_device()
    torch.manual_seed(0)
    H = 4 * C_M
    wp = {"norm_m.weight": torch.randn(C_M), "norm_m.bias": torch.randn(C_M),
          "norm_z.weight": torch.randn(C_Z), "norm_z.bias": torch.randn(C_Z),
          "proj_m.weight": torch.randn(NH * HD, C_M), "proj_g.weight": torch.randn(NH * HD, C_M),
          "proj_z.weight": torch.randn(NH, C_Z), "proj_o.weight": torch.randn(C_M, NH * HD)}
    wt = {"norm.weight": torch.randn(C_M), "norm.bias": torch.randn(C_M),
          "fc1.weight": torch.randn(H, C_M), "fc2.weight": torch.randn(H, C_M),
          "fc3.weight": torch.randn(C_M, H)}
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)
    pwa, tr = PairWeightedAveraging(HD, NH, wp, kc), Transition(wt, kc)
    print(f"\nfull update_msa probe: m=(1,{D},{S},{C_M}) chunk={CHUNK}")

    m_host = torch.randn(1, D, S, C_M)
    z_host = torch.randn(1, S, S, C_Z)

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dtype())

    z = up(z_host)

    m = up(m_host)                                    # --- whole branch
    m = ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
    whole_t = ttnn.to_torch(ttnn.add(m, ttnn.reshape(tr(m), tuple(m.shape))))

    m = up(m_host)                                    # --- chunked branch
    zc = ttnn.clone(z)
    parts = []
    for s in range(0, D, CHUNK):
        mc = m[:, s:min(s + CHUNK, D), :, :]
        mc = ttnn.add(mc, ttnn.reshape(pwa(mc, zc), tuple(mc.shape)))
        mc = ttnn.add(mc, ttnn.reshape(tr(mc), tuple(mc.shape)))
        parts.append(mc)
    chunked_t = ttnn.to_torch(ttnn.concat(parts, dim=1))

    # Positive control FIRST: prove the comparator can actually see a difference. Omitting this is
    # what let a broken comparison report vacuous "SAME" for two whole passes.
    # Perturb scale-aware: these are bf16, so "+1.0" on an element of magnitude ~1000 rounds
    # straight back to the same value and the control would fire spuriously (it did, first try).
    ctrl = chunked_t.clone()
    v = ctrl[0, 0, 0, 0].to(torch.float32).item()
    ctrl[0, 0, 0, 0] = v * 2.0 + 1.0 if v != 0.0 else 1.0
    assert not torch.equal(chunked_t, ctrl), "comparator is blind -- fix before trusting any verdict"
    assert not torch.equal(chunked_t, torch.zeros_like(chunked_t)), "output is all zeros"
    print("  [control] comparator detects a 1-element change   OK")

    report("full update_msa (PWA+Transition)", whole_t, chunked_t)


if __name__ == "__main__":
    main()

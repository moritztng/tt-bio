#!/usr/bin/env python3
"""L-A screen: is ESMFold2's diffusion head dispatch-bound, and would a captured trace pay?

esmfold2-to-4x.md labelled the head "dispatch bound over ~1.3k ops per step" without ever taking a
device-vs-host split. This takes it. Five numbers on the real per-step closure, at 512 aa:

  eager      dmm.step() with a device sync on both sides
  capture    the one-off cost of begin/record/end, paid once per fold
  traced     copy_host_to_device x2 + execute_trace + to_torch, synced both sides
  pure       execute_trace back to back, no host work, no sync between
  calls      ttnn calls per eager step, counted LAST so the counting wrapper never lands in a
             timed region

Kill gate, pre-committed in state/esmfold2-3p4x-close_PLAN.md sec 4:
  build the production path only if (eager - traced) * 68 >= 0.60 s, i.e. >= 8.8 ms/step.

Needs TT_BIO_TRACE_REGION_SIZE set (1 GiB) so the device opens with a trace region.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


class _Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.esmfold2 as E
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "esmfold2")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "esmfold2")

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "size": a.size, "reps": a.reps,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "trace_region_size": os.environ.get("TT_BIO_TRACE_REGION_SIZE"),
           "git_head": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
           "sampling_steps": B.SAMPLING_STEPS}

    tgt = a.fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = a.fixdir / ("cdk2x2_%d.a3m" % a.size)
    one_fold, meta, state = B.build_fold("esmfold2", ROOT / (".msa_ab512_%d" % a.size), tgt, a3m)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]

    grabbed = {}
    orig_ss = E.sample_structure

    def fake_ss(denoise_fn, n_atoms, ref_mask, **kw):
        fv = dict(zip(denoise_fn.__code__.co_freevars,
                      (c.cell_contents for c in denoise_fn.__closure__)))
        dmm, ft, head, sigma = fv["dmm"], fv["ft"], fv["self"], fv["sigma"]
        N = n_atoms
        Bsz = ref_mask.shape[0]

        # One representative step's host-side inputs, built exactly as `denoise` builds them.
        t = torch.full((Bsz,), float(sigma), dtype=torch.float32)
        t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
        n_raw_h = torch.cos(2.0 * 3.141592653589793 * (
            t_noise[:, None] * head._fw[None, :] + head._fb[None, :])).float().contiguous()
        x_noisy = torch.randn(Bsz, N, 3, generator=torch.Generator().manual_seed(0)) * sigma
        denom = torch.sqrt(t * t + sigma * sigma)
        r_noisy = x_noisy / denom[:, None, None]
        r_input_h = torch.cat([r_noisy, torch.zeros_like(r_noisy)], dim=-1).float().contiguous()

        n_dev, r_dev = ft(n_raw_h), ft(r_input_h)

        def sync():
            ttnn.synchronize_device(dev)

        # 1. eager
        for _ in range(2):
            ttnn.deallocate(dmm.step(n_dev, r_dev))
        sync()
        eager = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            out = dmm.step(n_dev, r_dev)
            sync()
            eager.append((time.perf_counter() - t0) * 1e3)
            eager_ref = ttnn.to_torch(out).clone()
            ttnn.deallocate(out)

        # 2. capture
        sync()
        t0 = time.perf_counter()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        traced_out = dmm.step(n_dev, r_dev)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        sync()
        capture_ms = (time.perf_counter() - t0) * 1e3

        # 3. traced, host round trip included
        host_n = ttnn.from_torch(n_raw_h, layout=ttnn.TILE_LAYOUT, dtype=n_dev.dtype)
        host_r = ttnn.from_torch(r_input_h, layout=ttnn.TILE_LAYOUT, dtype=r_dev.dtype)
        for _ in range(2):
            ttnn.copy_host_to_device_tensor(host_n, n_dev)
            ttnn.copy_host_to_device_tensor(host_r, r_dev)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            _ = ttnn.to_torch(traced_out)
        sync()
        traced = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(host_n, n_dev)
            ttnn.copy_host_to_device_tensor(host_r, r_dev)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            got = ttnn.to_torch(traced_out)
            sync()
            traced.append((time.perf_counter() - t0) * 1e3)
        traced_ref = got.clone()

        # 4. pure device: N executes back to back, one sync at the end
        sync()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        sync()
        pure_ms = (time.perf_counter() - t0) * 1e3 / a.reps

        # 5. parity of the traced output against the eager one
        res["parity_torch_equal"] = bool(torch.equal(
            torch.Tensor(eager_ref).float(), torch.Tensor(traced_ref).float()))
        res["parity_max_abs"] = float(
            (torch.Tensor(eager_ref).float() - torch.Tensor(traced_ref).float()).abs().max())

        # 6. call count, installed LAST so it never lands in a timed region
        counted = [0]
        import types
        wrapped = []
        for name in dir(ttnn):
            fn = getattr(ttnn, name, None)
            if callable(fn) and not isinstance(fn, type) and not name.startswith("_"):
                try:
                    def mk(f):
                        def c(*A, **K):
                            counted[0] += 1
                            return f(*A, **K)
                        return c
                    setattr(ttnn, name, mk(fn))
                    wrapped.append((name, fn))
                except Exception:
                    pass
        try:
            counted[0] = 0
            ttnn.deallocate(dmm.step(n_dev, r_dev))
            calls = counted[0]
        finally:
            for name, fn in wrapped:
                setattr(ttnn, name, fn)

        res.update(
            N_atoms=int(N), B=int(Bsz),
            eager_ms=round(st.median(eager), 4), eager_all=[round(v, 3) for v in eager],
            traced_ms=round(st.median(traced), 4), traced_all=[round(v, 3) for v in traced],
            capture_ms=round(capture_ms, 3), pure_execute_ms=round(pure_ms, 4),
            ttnn_calls_per_step=calls,
        )
        try:
            ttnn.release_trace(dev, tid)
        except Exception:
            pass
        raise _Stop()

    E.sample_structure = fake_ss
    try:
        one_fold()
    except _Stop:
        pass
    finally:
        E.sample_structure = orig_ss

    steps = 68
    d_ms = res["eager_ms"] - res["traced_ms"]
    res["steps_executed"] = steps
    res["fold_delta_s"] = round(d_ms * steps / 1e3 - res["capture_ms"] / 1e3, 4)
    res["kill_gate_ms_per_step"] = 8.8
    res["GO"] = bool(d_ms >= 8.8)
    res["device_share_pct"] = round(100.0 * res["pure_execute_ms"] / res["eager_ms"], 2)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_all")}, indent=1))


if __name__ == "__main__":
    main()

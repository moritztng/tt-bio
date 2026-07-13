"""Per-step device parity for OpenDDE diffusion trace replay.

OpenDDE's diffusion sampler is the shared Protenix-v2 EDM loop (``edm_sample`` ->
``DiffusionModule.denoise``), so an end-to-end trace-on vs trace-off coord diff
cannot isolate the trace path: ttnn reduction nondeterminism means two untraced
folds already diverge run-to-run (see the BoltzGen precedent,
``perf/boltzgen_trace_step_parity/``). Instead this harness proves the device-
level claim directly: on the FIRST per-step denoise of a real OpenDDE fold, run
BOTH ``denoise`` (untraced) and ``denoise_traced`` on the identical
``(x_noisy, t_hat, cond)`` -- same weights, same resident cache -- and compare
the returned coords bit-for-bit. Trace replay reuses the exact captured device
program with new input buffer contents, so this must be 0.

Run with a trace region reserved and the fold itself traced:

    TT_VISIBLE_DEVICES=1 TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto \
      TT_BIO_TRACE_REGION_SIZE=1073741824 \
      PYTHONPATH=$PWD/perf/opendde_trace_step_parity:$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m perf.opendde_trace_step_parity.run

Writes /tmp/trace_parity_maxdiff.txt (0.0 = bit-identical).
"""
import os

import torch


def _install():
    try:
        import tt_bio.protenix as P
    except Exception:
        return
    DiffusionModule = P.DiffusionModule
    orig_denoise = DiffusionModule.denoise
    orig_denoise_traced = DiffusionModule.denoise_traced
    state = {"done": False}

    def denoise_traced(self, x_noisy, t_hat, cond):
        if not state["done"]:
            state["done"] = True
            try:
                off = orig_denoise(self, x_noisy, t_hat, cond)
                on = orig_denoise_traced(self, x_noisy, t_hat, cond)
                ro = off.float()
                rn = on.float()
                md = float((ro - rn).abs().max())
                exact = bool(torch.equal(ro, rn))
                msg = (f"[TRACE_PARITY] per-step denoise maxdiff={md} "
                       f"exact={exact} shape={tuple(ro.shape)}")
                print(msg, flush=True)
                with open("/tmp/trace_parity_maxdiff.txt", "w") as f:
                    f.write(f"{md}\nexact={exact}\n{msg}\n")
            except Exception as e:
                print(f"[TRACE_PARITY] error: {e!r}", flush=True)
                with open("/tmp/trace_parity_maxdiff.txt", "w") as f:
                    f.write(f"ERR {e!r}\n")
        return orig_denoise_traced(self, x_noisy, t_hat, cond)

    DiffusionModule.denoise_traced = denoise_traced


_install()

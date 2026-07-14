"""Per-step device parity for BoltzGen diffusion trace replay.

The end-to-end design forward is NOT bit-deterministic run-to-run on ttnn
(reduction order in the shared Pairformer/DiT drifts ~10 A over 500 steps even
between two untraced runs), so an end-to-end trace-on vs trace-off coord
comparison cannot isolate the trace path. Instead this harness proves the
device-level claim directly: on the FIRST per-step score-model call of a real
design, run BOTH ``forward(trace=False)`` and ``forward(trace=True)`` on the
identical (r_noisy, times, conditioning) inputs — same weights, same resident
cache — and compare ``r_update`` bit-for-bit. Trace replay reuses the exact
captured device program with new input buffer contents, so this must be 0.

Run with a trace region reserved but the run itself untraced:

    TT_VISIBLE_DEVICES=3 TT_MESH_GRAPH_DESC_PATH=... TT_BIO_TRACE_REGION_SIZE=1073741824 \
      PYTHONPATH=$PWD/perf/boltzgen_trace_step_parity:$PWD \
      python3 -m tt_bio.main gen run examples/binder.yaml --output /tmp/bg_step \
        --num_designs 1 --devices 1 --budget 1 --steps design

Reads /tmp/trace_parity_maxdiff.txt (0.0 = bit-identical).
"""
import os

import torch


def _install():
    try:
        import tt_bio.boltzgen.adapter as A
    except Exception:
        return
    TTScoreModelAdapter = A.TTScoreModelAdapter
    orig_forward = TTScoreModelAdapter.forward
    state = {"done": False}

    def forward(self, *, r_noisy, times, s_inputs, s_trunk, feats,
                diffusion_conditioning, multiplicity=1, trace=False, **_u):
        if not state["done"]:
            state["done"] = True
            try:
                off = orig_forward(
                    self, r_noisy=r_noisy, times=times, s_inputs=s_inputs,
                    s_trunk=s_trunk, feats=feats,
                    diffusion_conditioning=diffusion_conditioning,
                    multiplicity=multiplicity, trace=False)
                on = orig_forward(
                    self, r_noisy=r_noisy, times=times, s_inputs=s_inputs,
                    s_trunk=s_trunk, feats=feats,
                    diffusion_conditioning=diffusion_conditioning,
                    multiplicity=multiplicity, trace=True)
                ro = off["r_update"].float()
                rn = on["r_update"].float()
                md = float((ro - rn).abs().max())
                exact = bool(torch.equal(ro, rn))
                msg = (f"[TRACE_PARITY] per-step r_update maxdiff={md} "
                       f"exact={exact} shape={tuple(ro.shape)}")
                print(msg, flush=True)
                with open("/tmp/trace_parity_maxdiff.txt", "w") as f:
                    f.write(f"{md}\nexact={exact}\n{msg}\n")
            except Exception as e:
                print(f"[TRACE_PARITY] error: {e!r}", flush=True)
                with open("/tmp/trace_parity_maxdiff.txt", "w") as f:
                    f.write(f"ERR {e!r}\n")
        return orig_forward(
            self, r_noisy=r_noisy, times=times, s_inputs=s_inputs,
            s_trunk=s_trunk, feats=feats,
            diffusion_conditioning=diffusion_conditioning,
            multiplicity=multiplicity, trace=trace)

    TTScoreModelAdapter.forward = forward


_install()

"""Per-step device parity for Boltz-2 diffusion trace replay.

The end-to-end Boltz-2 fold is NOT bit-deterministic run-to-run on ttnn
(reduction order in the shared Pairformer/DiT drifts across the 200 sampling
steps even between two untraced runs), so an end-to-end trace-on vs trace-off
coord comparison cannot isolate the trace path — see the same finding in
``perf/boltzgen_trace_step_parity/sitecustomize.py``. Instead this harness
proves the device-level claim directly: on the FIRST per-step score-model call
of a real fold, run BOTH ``DiffusionModule.forward`` (untraced) and
``DiffusionModule.forward_traced`` on the identical ``(r, times, conditioning)``
inputs — same weights, same resident cache — and compare the returned
``r_update`` bit-for-bit. Trace replay reuses the exact captured device program
with new input buffer contents, so this must be 0.

Run with a trace region reserved but the fold itself untraced (so the only
exerciser of forward_traced is this harness):

    TT_VISIBLE_DEVICES=0 \\
      TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto \\
      TT_BIO_TRACE_REGION_SIZE=1073741824 \\
      PYTHONPATH=$PWD/perf/boltz2_trace_step_parity:$PWD \\
      python3 -m tt_bio.main predict examples/prot.yaml --model boltz2 \\
        --single_sequence --sampling_steps 5 --diffusion_samples 1 --seed 0 \\
        --accelerator tenstorrent --out_dir /tmp/b2_step_parity

Reads /tmp/trace_parity_maxdiff_boltz2.txt (0.0 = bit-identical).
"""
import os

import torch


def _install():
    try:
        import tt_bio.tenstorrent as T
    except Exception:
        return
    DiffusionModule = T.DiffusionModule
    orig_forward = DiffusionModule.forward
    state = {"done": False}

    def forward(self, r, times, s_inputs, s_trunk, q, c,
                bias_encoder, bias_token, bias_decoder,
                keys_indexing, mask, atom_to_token):
        if not state["done"]:
            state["done"] = True
            try:
                off = orig_forward(
                    self, r, times, s_inputs, s_trunk, q, c,
                    bias_encoder, bias_token, bias_decoder,
                    keys_indexing, mask, atom_to_token)
                on = self.forward_traced(
                    r, times, s_inputs, s_trunk, q, c,
                    bias_encoder, bias_token, bias_decoder,
                    keys_indexing, mask, atom_to_token)
                ro = off.float()
                rn = on.float()
                md = float((ro - rn).abs().max())
                exact = bool(torch.equal(ro, rn))
                msg = (f"[TRACE_PARITY_BOLTZ2] per-step r_update maxdiff={md} "
                       f"exact={exact} shape={tuple(ro.shape)}")
                print(msg, flush=True)
                with open("/tmp/trace_parity_maxdiff_boltz2.txt", "w") as f:
                    f.write(f"{md}\nexact={exact}\n{msg}\n")
                # release the trace we captured so it doesn't leak device memory
                try:
                    self._release_trace()
                except Exception:
                    pass
            except Exception as e:
                print(f"[TRACE_PARITY_BOLTZ2] error: {e!r}", flush=True)
                with open("/tmp/trace_parity_maxdiff_boltz2.txt", "w") as f:
                    f.write(f"ERR {e!r}\n")
        return orig_forward(
            self, r, times, s_inputs, s_trunk, q, c,
            bias_encoder, bias_token, bias_decoder,
            keys_indexing, mask, atom_to_token)

    DiffusionModule.forward = forward


_install()

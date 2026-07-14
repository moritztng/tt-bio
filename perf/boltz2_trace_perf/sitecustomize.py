"""Warm wall-clock of Boltz-2 diffusion (AtomDiffusion.sample) — trace ON vs OFF.

Times the full diffusion sampling loop (one ``sample()`` call covers all
diffusion_samples x sampling_steps) and counts ``forward_traced`` vs untraced
``forward`` calls to confirm the trace path is actually exercised when
``--diffusion_trace`` is on. Run the same fold twice (trace off, trace on) and
read /tmp/boltz2_diffusion_seconds.txt:

    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... \\
      PYTHONPATH=$PWD/perf/boltz2_trace_perf:$PWD \\
      python3 -m tt_bio.main predict examples/prot.yaml --model boltz2 \\
        --single_sequence --sampling_steps 200 --diffusion_samples 1 --seed 0 \\
        --accelerator tenstorrent [--diffusion_trace] --out_dir /tmp/b2_perf

Within one process the first diffusion step compiles (cold) and, with trace on,
also captures the trace; the remaining 199 steps are warm steady-state, so the
total / num_steps is dominated by the warm per-step time (the headline number).
"""
import time

import torch


def _install():
    try:
        import tt_bio.boltz2 as B
        import tt_bio.tenstorrent as TT
    except Exception:
        return
    AtomDiffusion = B.AtomDiffusion
    orig_sample = AtomDiffusion.sample
    counts = {"traced": 0, "untraced": 0}
    DiffusionModule = TT.DiffusionModule
    orig_traced = DiffusionModule.forward_traced
    orig_forward = DiffusionModule.forward

    def traced(self, *a, **k):
        counts["traced"] += 1
        return orig_traced(self, *a, **k)

    def forward(self, *a, **k):
        counts["untraced"] += 1
        return orig_forward(self, *a, **k)

    DiffusionModule.forward_traced = traced
    DiffusionModule.forward = forward

    state = {"secs": None, "n_steps": None, "n_samples": None}

    def sample(self, *a, **k):
        n_steps = k.get("num_sampling_steps") or self.num_sampling_steps
        n_samples = k.get("diffusion_samples", 1)
        t0 = time.monotonic()
        out = orig_sample(self, *a, **k)
        dt = time.monotonic() - t0
        state["secs"] = dt
        state["n_steps"] = n_steps
        state["n_samples"] = n_samples
        per_step = dt / max(1, n_steps * n_samples)
        msg = (f"[B2_DIFF_PERF] sample()={dt:.3f}s steps={n_steps} "
               f"samples={n_samples} per_step={per_step:.4f}s "
               f"traced={counts['traced']} untraced={counts['untraced']}")
        print(msg, flush=True)
        with open("/tmp/boltz2_diffusion_seconds.txt", "w") as f:
            f.write(msg + "\n")
        return out

    AtomDiffusion.sample = sample


_install()

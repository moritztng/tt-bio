"""What RFD4's four readouts cost the host per sampler step, at PopVax's width.

`FlowGlue.heads` computes all four every step. The sampler keeps `bb_ca` and `local_latents` from
every step, `sequence` only on a trajectory snapshot, and `insertion` only from the final step
(`tt_bio/rfd4/sampler.py` SamplerOutput: "from the final step"). The insertion head is two [n, d, d]
projections and an [n, n, d] bmm, so it grows with n^2 while the kept heads grow with n.

Card-free: it only needs the checkpoint's head weights.

    python perf/rfd4/heads/cost.py --weights ~/rfd4-bundle --n 620 --reps 20
"""
import argparse
import json
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=620)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--steps", type=int, default=60, help="sampler steps a design, for the per-design total")
    ap.add_argument("--out")
    a = ap.parse_args()

    from tt_bio.rfd4 import network as N

    state, config = N.load_checkpoint(a.weights)
    glue = N.FlowGlue(state, config)
    tok = config["nn"]["token_dim"]
    s = torch.randn(a.batch, a.n, tok)
    mask = torch.ones(a.batch, a.n)

    def med(fn):
        fn()
        ms = []
        for _ in range(a.reps):
            t = time.perf_counter()
            fn()
            ms.append(1e3 * (time.perf_counter() - t))
        return round(statistics.median(ms), 3)

    with torch.no_grad():
        parts = {
            "ca_linear": med(lambda: glue.ca_linear(s)),
            "local_latents_linear": med(lambda: glue.local_latents_linear(s)),
            "insertion": med(lambda: torch.bmm(glue.insertion_q_proj(s),
                                               glue.insertion_k_proj(s).transpose(1, 2))),
        }
        if glue.sequence_readout_enabled:
            parts["sequence_readout"] = med(lambda: glue.sequence_readout(s))
        batch = {"padding_mask": mask, "x_t": {}, "asym_id": torch.zeros(a.batch, a.n, dtype=torch.long)}
        try:
            whole = med(lambda: glue.heads(s, batch))
        except Exception as exc:           # emit_masks needs the real featurized batch
            whole = None
            parts["heads_note"] = f"whole-heads call needs a featurized batch: {exc.__class__.__name__}"

    rec = {"n": a.n, "batch": a.batch, "token_dim": tok, "threads": torch.get_num_threads(),
           "ms_median": parts, "heads_ms_median": whole, "steps": a.steps,
           "insertion_s_per_design": round(parts["insertion"] * a.steps / 1e3, 3),
           "insertion_s_wasted_per_design": round(parts["insertion"] * (a.steps - 1) / 1e3, 3)}
    print(json.dumps(rec, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()

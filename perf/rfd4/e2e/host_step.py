"""The host half of one RFD4 design, card-free: every step's glue and sampler work with the trunk stubbed.

    python perf/rfd4/e2e/host_step.py --spec <spec.yaml> --weights <bundle> [--designs 2] [--out x.jsonl]

The trunk is replaced by a stub that returns a fixed `s`, so what is timed is exactly the host
work the device path does around a trunk call (pair-on-card form: the host sends bins, scales and
the AdaLN modulation, not the pair). Each piece is labelled by what it reads:

* x-independent (`cond_s`, `pair_mod_s`): the conditioning transitions and the pair AdaLN's
  gamma/beta read only `t` and the remaining-pass count, so step k+1's are known before step k's
  trunk returns;
* x-dependent (`init_s`, `pair_bins_s`, `heads_s`, `sampler_s`): need the trunk's output or the
  state it moves.

`featurize_s` (once per input), `constant_s` (the pair constant, once per design) and `write_s`
(the CIF writer on the stub's output) are the per-design host pieces outside the step loop.
Device-side host work (pad, upload, readback) needs a card and is not here.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--designs", type=int, default=2, help="the first is a warm-up")
    ap.add_argument("--out")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks
    from tt_bio.rfd4 import engine as E, network as N, sampler as S

    tmp = Path("/tmp/host_step") / Path(a.spec).stem
    designs = rfd4.prepare_examples(a.spec, tmp / "inputs")
    t0 = time.perf_counter()
    features, atom_array = rfd4.featurize(designs[0].cif, seed=0)
    featurize_s = time.perf_counter() - t0
    state, config = N.load_checkpoint(Path(a.weights).expanduser())
    glue = N.FlowGlue(state, config)
    spec = S.recycle_strategy("thinking_medium")
    batch = E.preprocess(features)
    n = batch["padding_mask"].shape[1]
    acc = defaultdict(float)

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            r = fn(*args, **kwargs)
            acc[name] += time.perf_counter() - t
            return r
        return wrapper

    split = glue.pair_split
    glue.init_repr_factory.forward = timed("init_s", glue.init_repr_factory.forward)
    glue.conditioning = timed("cond_s", glue.conditioning)
    glue.heads = timed("heads_s", glue.heads)
    split.modulation = timed("pair_mod_s", split.modulation)
    split.bins = timed("pair_bins_s", split.bins)
    split.constant = timed("constant_s", split.constant)

    s_fixed = torch.randn(1, n, glue.cfg["token_dim"], generator=torch.Generator().manual_seed(0)) * 0.1

    class StubTrunk:
        builds_pair = True

        def launch(self, s_init, z_init, conds, mask):
            return s_fixed.expand(s_init.shape[0], -1, -1)

        def collect(self, pending):
            return pending, None

    class Out:
        pass

    rows = []
    for i in range(a.designs):
        acc.clear()
        net = N.FlowNetwork(glue, StubTrunk(), batch)
        for stage in ("prepare", "launch", "finish"):
            setattr(net, stage, timed(f"{stage}_total_s", getattr(net, stage)))
        load0 = os.getloadavg()[0]
        t0 = time.perf_counter()
        out = S.generate(spec, net, n_designs=1, n_tokens=n, seed=i, rng_mode="per_design",
                         emit_masks=compute_flow_emit_masks(batch), groups=E._groups(batch),
                         coord_cond_mask=batch.get("coord_cond_mask"))
        loop_s = time.perf_counter() - t0
        # The CIF writer on a stand-in output with the real shapes (CA at the sampled positions).
        o = Out()
        o.coords_nm = torch.zeros(1, n, 37, 3)
        o.coords_nm[:, :, 1] = out.x["bb_ca"]
        o.residue_type = torch.zeros(1, n, dtype=torch.long)
        o.atom_mask = torch.zeros(1, n, 37, dtype=torch.bool)
        o.atom_mask[:, :, :4] = True
        o.insertion_logits = out.insertion_logits
        t1 = time.perf_counter()
        try:
            rfd4.write_design(o, features, atom_array, tmp / f"d{i}.cif")
            write_s = time.perf_counter() - t1
        except Exception as exc:  # noqa: BLE001 -- a stand-in output may not satisfy the writer
            write_s = None
            print("write skipped:", exc, file=sys.stderr)
        pieces = {k: round(v, 3) for k, v in sorted(acc.items())}
        sampler_s = loop_s - acc["prepare_total_s"] - acc["launch_total_s"] - acc["finish_total_s"]
        rec = {"size": n, "design": i, "kind": "warmup" if i == 0 else "warm", "steps": spec.nsteps,
               "loop_s": round(loop_s, 3), **pieces, "sampler_s": round(sampler_s, 3),
               "x_independent_s": round(acc["cond_s"] + acc["pair_mod_s"], 3),
               "featurize_s": round(featurize_s, 3), "write_s": None if write_s is None else round(write_s, 3),
               "threads": torch.get_num_threads(), "load1": round(load0, 2),
               "load1_end": round(os.getloadavg()[0], 2), "nproc": os.cpu_count(),
               "host": os.uname().nodename, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

# perf/esm512 — ESMFold2 512 aa deep perf, Phase 1 + 2

Analysis and verdict live in `~/.coworker/state/esmfold2-512aa-deep-perf.md`. This directory holds
the three harnesses and their raw output on qb2 card 2, ttnn 0.68.0, all under `benchlock.sh`.

* `decomp.py` — the fold wall plus an inclusive/exclusive component decomposition. Timers are
  esmfold2s own classes: the shared `TriangleAttention` is never constructed for this model
  (`esmfold2.py:28-32` imports only `TriangleMultiplication`), so the protenix-family timer set
  would install onto classes that never run. `--plain` folds are untimed and are the number that
  counts; `--timed` folds carry the attribution and cost +0.75 %.
* `roofs.py` — DRAM and matmul roofs measured on the card, at the fidelity and core grid the model
  actually runs (HiFi4, fp32_dest_acc_en, 11x10). Nothing here is inherited.
* `screen.py` — the pair transition priced op by op at the production shape, then each candidate
  rewrite run end to end and checked with `torch.equal`, plus the E6 gate on the real trimul shape.

Headline results, all measured:

    fold 45.816 s (median of 4, A/A spread 0.156 s)
    block:PairUpdateBlock 36.917 s = 80.6 % of the fold
    DRAM add roof 431.1 GB/s, matmul 4096^3 bf16 HiFi4 112.7 TFLOP/s
    E6 on the trimul       18.135 -> 14.765 ms, 18 gated moves served, 0 declined
    pair transition        29.405 -> 21.147 ms with the split-fc1 rewrite, torch.equal True
    SwiGLUFFN.fuse_swiglu  False on this wheel: minimal_matmul has no fuse_swiglu kwarg

Reproduce:

    benchlock.sh esmfold2-512aa-deep-perf -- env TT_VISIBLE_DEVICES=2 \
      TT_BIO_LEASE_HOLDER=worker:esmfold2-512aa-deep-perf PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 perf/esm512/screen.py --L 512 \
      --out perf/esm512/screen_512_c2.json

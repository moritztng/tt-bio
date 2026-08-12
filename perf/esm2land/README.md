# perf/esm2land — what the stranded ESMFold2 pair-FFN change actually does

Plan and verdict: `~/.coworker/state/esmfold2-l1-pair-ffn-extract-land.md`.

`probe_parity.py` prices the pair transition at several sequence lengths and, for every arm, records
which of `_pair_proj_linear`'s four legs each call took and whether the result is `torch.equal`
against the chain main ships today. It exists because the change on `wk/esmfold2-to-4x` was verified
at 512 aa only, and both its parity and its speedup turn out to depend on the length.

Measured on qb2 card 2, ttnn 0.68.0, c_z=256, d_ff=1024, grid 11x10 (`probe_parity_c2.json`):

    arm                          298      320      512     exact 298/320/512
    ref_shipped (main)          9.564   10.196   29.442    --
    A  split fc1 + SiLU         6.860    7.314   21.166    True  / True / True
    C  branch row-block + L1    6.778    7.285   18.067    False / True / True
    E  row-block, plain _lin,
       L1 SwiGLU product        6.767    7.297   18.117    False / True / True
    F  same, DRAM product       7.984    8.515   21.407    False / True / True

Three things follow. The L1 leg the branch is named for never fires: every call at every length takes
`L1_cfg_None -> core_grid_untuned`, the same kernel main already runs, and `_L1_OUT_REFUSED` stays
empty because the allocator is never asked. E matches C, so that leg is inert and can be dropped. F
is a loss everywhere, so the win is the `memory_config=L1` on the SwiGLU product that fc2 reads, and
the row block is only what makes that product small enough to be served.

Parity needs the block height to divide the length. At 298 the chunk is nine rows of 30 plus a tail
of 28, `ttnn.linear` derives a different program for the tail, and the result stops being
`torch.equal`. 320 and 512 divide by 32 and are exact.

Reproduce:

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 perf/esm2land/probe_parity.py \
      --L 298 320 512 --rows 32 --out perf/esm2land/probe_parity_c2.json

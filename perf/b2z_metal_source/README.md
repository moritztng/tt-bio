# b2z-metal-source-recon — two experiments a carded sibling can run

Source recon of tt-metal `v0.68.0` (`1452925b`) and its pinned LLK `tt-llk` `a7a846ae`, asking
whether the Boltz-2 deficit is in tt-metal and whether it needs a fork. Memo:
`~/.coworker/state/b2z/FORK-CASE.md`. Short answer: no fork — everything that matters ships as
text in the wheel and is JIT-compiled at runtime.

Neither script has been run. Both need a card; this row had none.

## `make_overlay.py` — patch the LLK without forking or rebuilding

`tt_metal/llrt/rtoptions.cpp:216` reads `TT_METAL_RUNTIME_ROOT`, which repoints the whole JIT
source tree. So an LLK change is a text edit plus an env var.

The edit: on Blackhole, `program_packer_destination`
(`tt_llk_blackhole/common/inc/cpack_common.h:536`) takes a `TTI_STALLWAIT(STALL_CFG, THCON)` before
its `WRCFG` on **every packed tile**. Wormhole's copy of the same function has that stall commented
out and uses `TTI_REG2FLOP` instead. Two arms test whether Blackhole needs it:

```
python3 make_overlay.py --arm none      # control
python3 make_overlay.py --arm no_stall  # drop the stall, keep WRCFG
python3 make_overlay.py --arm reg2flop  # Wormhole's sequence verbatim
export TT_METAL_RUNTIME_ROOT=/tmp/b2z-overlay-<arm>
rm -rf ~/.cache/tt-metal-cache*         # or the old kernel binaries are reused
```

The overlay is hard-linked, so it costs inodes and one file, not 180 MB. The script refuses to run
unless `cpack_common.h` hashes to the pinned `a7a846ae` content.

Address programming only, so **the fold must stay bit-exact**. A wrong or hanging fold is the
falsification, and it is a cheap one. Predicted 1.02-1.10x on a pack-heavy op; falsifier is < 1 % on
`b2z-custom-sdpa`'s SDPA microbenchmark, which already has an A/A floor.

## `welford_ab.py` — the config-only lever nobody has tried

tt-metal's default layernorm kernel makes five full-width pack passes over every row
(`layernorm.cpp:186, :214, :269, :305, :332`) where the mathematics needs one. LayerNorm is 10.1 %
of the Pairformer block, 1.032 s/fold, and tt-bio passes no program config to `ttnn.layer_norm`
anywhere, so every call runs the five-pass kernel.

`use_welford` (`layernorm_device_operation.cpp:215`, bound at `layernorm_nanobind.cpp:45`) removes
**one** of the five — counting the passes in `layernorm_welford.cpp` (`:216, :272, :300, :328`), it
folds away `cb_xmm2` and nothing else. So expect ~1.005x, not a large win. It is worth running
because it is free and because it bounds how much of this op the packer actually prices; the real
lever is folding gamma and beta into the normalize pass (P1b in the memo), which needs a kernel edit
through the same overlay.

```
python3 welford_ab.py --device-id 0 --seq 512 --dim 128
```

Interleaved A/B in one process with an A/A floor. Welford changes the reduction order, so this is
**not** bit-exact and a win has to clear the `cdk2x2_298` structural control before it goes near
main.

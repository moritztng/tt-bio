# Does BindCraft 2's optimiser see the same molecule its filters do?

`bindcraft/af2.py:308` renumbers a multi-chain complex before a monomer AlphaFold2 sees it, so the
model does not read the two chains as one continuous peptide. The gradient path at
`af2.py:340-348` builds the same features and never makes that call. This directory measures what
the difference costs on the metrics BindCraft 2's own filters read.

Everything here is upstream BindCraft 2 on CPU. No Tenstorrent anything, no tt-bio import, no GPU.

## Running it

From a clean checkout of BindCraft 2 (measured at `7a2dfdb`, re-verified at `301efdd`):

    export PYTHONPATH=/path/to/BindCraft2
    python chainbreak.py census
    python chainbreak.py run --leg predict          --binder 32 --target 96 --params /path/to/af2_params
    python chainbreak.py run --leg predict_identity --binder 32 --target 96 --params /path/to/af2_params
    python chainbreak.py run --leg gradient         --binder 32 --target 96 --params /path/to/af2_params

`census` needs no model and no parameters and finishes in under a second.

The model legs need one file, `params_model_1_ptm.npz`, the monomer AlphaFold2 parameter set that
`bindcraft fetch-weights` downloads. `--params` points at the directory holding it.

Each leg runs in its own process on purpose. Both entry points memoise their traced function
(`prediction_compile_cache`, `gradient_compile_cache`) and neither cache key mentions the residue
numbering, so a patch installed inside a process that has already traced returns the old answer
bit-for-bit and looks like a null result.

## What the legs are

- `predict` — `AlphaFoldDesignModel.predict`, the path the mutate and final filters use.
- `predict_identity` — the same call with `monomer_chain_break_indices` replaced by the identity,
  which is the numbering the gradient path passes.
- `gradient` — `AlphaFoldDesignModel.sequence_gradients`, the path the screen, refine, anneal and
  harden stages are graded on.

Both chains are built from sequence, so no structure file is needed. Chain lengths are chosen as
multiples of 32 so that neither path's bucket padding fires and the residue numbering is the only
input that differs between them.

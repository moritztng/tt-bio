`monomer_chain_break_indices` (`bindcraft/af2.py:185`) has one call site: `_predict_complex` at
`af2.py:308`. The gradient path, `predict_complex_arrays` inside `_compiled_sequence_gradients`
(`af2.py:340-348`), builds `asym_id`, `entity_id` and `seq_mask` with the same three helpers
`_predict_complex` uses, then passes `residue_index` straight into `alphafold_input_features`.

For a monomer model that is the whole of chain identity. Monomer relpos reads `offset` clipped to
+-32 (`af/alphafold/model/modules.py:1469-1484`, `max_relative_feature` 32) and `asym_id` never
reaches it. So with `model_1_ptm` on a two-chain complex, `predict` and `sequence_gradients` hand
AlphaFold2 different molecules: the gradient path sees the binder and the target as one continuous
peptide, which is exactly what the docstring on `monomer_chain_break_indices` says must not happen.

### The difference is that one call, bit for bit

Binder 32 + target 96 built from sequence, `model_1_ptm`, `num_recycle=1`, `dropout=False`,
identical sequence parameters on both sides, CPU, one leg per process:

| entry point | pTM | i_pTM | pLDDT |
|---|---|---|---|
| `predict` | 0.294875 | 0.141805 | 0.363348 |
| `sequence_gradients` | 0.352421 | **0.522476** | 0.408438 |
| `predict`, with `monomer_chain_break_indices` replaced by the identity | 0.352421 | **0.522476** | 0.408438 |

Rows 2 and 3 agree on all four metrics **to the last bit** (`0.35242122411727905`,
`0.5224756002426147`, `0.4084381478605792`, `0.44594516418874264`), in separate processes. The
reverse closes too: feeding `sequence_gradients` a target whose own numbering already carries the
junction step of 50 reproduces row 1 bit for bit.

Two controls, both bit-exact:

- **One chain.** The branch at `:308` cannot fire, and the two entry points agree — pTM identical,
  pLDDT 8e-9 apart, which is float32 reduction order.
- **Zero pairs in the window.** Renumber the target so no cross-chain pair is within +-32 and the
  two numberings give the same answer to the last bit. The carrier is the relpos window, not the
  numbering as such.

### Where it lands in a trajectory

- `run_gradient_design_stage` takes its predictions from `sequence_gradients`
  (`trajectory.py:131`), and `judge_stage` grades `screen`, `refine`, `anneal` and `harden` on
  those same objects (`:310`). For a single-target campaign nothing replaces them: the pooling that
  would call `predict` is gated on `len(prepared_states) > 1` (`:178-183`).
- `run_sequence_mutation_stage` scores every candidate with `predict` (`:162`), and `mutate` is
  graded on `predict` (`:272-275`). `final` inherits those predictions (`:321-323`).

So a single-target trajectory optimises against one numbering for four stages and against the other
at mutate, and four of its six gates read a different instrument from the last two.

### How big, across binder draws

Gap is the gradient path's numbering minus the predict path's, same complex, one row per arm:

PLACEHOLDER_TABLE

The damage sits almost entirely in i_pTM; the pTM and pLDDT gaps are an order of magnitude smaller
and change sign between arms. i_pTM is what the stage filters reject on.

The size is set by the **shorter** chain, not the target. The count of cross-chain pairs inside the
+-32 window is about `65 * binder_length` once the target is longer than the binder, so it is flat
across the three target lengths above (1552 in every row) while the share falls from 50% to 17%.
The gap follows the count, not the share. We expected the opposite before running it and were
wrong.

### Which way to fix it is yours

Two readings fit: the gradient path is missing the call, or `_predict_complex` adds it
deliberately. We have no view on which you intend.

One argument that might favour leaving it out does not survive: there is no differentiability cost.
`sequence_design_loss` is differentiated with `jax.value_and_grad(..., argnums=2)`
(`af2.py:378`), and argnum 2 is `sequences`; `residue_index` arrives in `state_templates`, argnum 3,
which is not differentiated. We also ran the gradient path on the chain-broken numbering directly:
it compiles and returns a finite design loss and finite sequence gradients.

Happy to send a patch once you say which side should move.

### Reproducer

Pure upstream BindCraft 2 on CPU, no GPU and no structure file — both chains are built from
sequence. It needs `PYTHONPATH` pointed at the checkout and `params_model_1_ptm.npz` from
`bindcraft fetch-weights`. `census` needs neither the model nor the parameters. On 32 CPU cores a
`predict` leg at 128 residues takes about a minute including compile and the `sequence_gradients`
leg about three.

Each leg must run in its own process: both entry points memoise their traced function
(`prediction_compile_cache`, `gradient_compile_cache`) and neither cache key mentions the residue
numbering, so a patch applied inside a process that has already traced returns the old answer
bit-for-bit and reads as a null result.

PLACEHOLDER_SCRIPT

Measured at `7a2dfdb`, re-read at `301efdd` where both the single call site and the stage/filter
split are unchanged.

Not claiming this is the cause of #13 — that was closed by the `aa_bias` fix in #17, and we have
not tested this against the PDL1-VHH case. It is in the same area and still live at HEAD, so it
seemed worth reporting separately.

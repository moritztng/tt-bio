# D19 — where the forward gap is not

`of3t-gradients` measured the same step at `num_recycles = 0` with every recorded draw pinned and
got loss **1.6311432393241485** on qb2 against the reference's **1.6422035029890711**, a 6.73e-03
relative disagreement. A gradient comparison cannot be tighter than the forward it is taken at, so
that is a floor under instrument A and it has to be attributed.

The suspects were upstream's two downcasts, both of which are precision floors in their shipped
bf16/fp32 paths and become downcasts inside a float64 model:

- `torch.amp.autocast(device_type=..., dtype=torch.float32)` around their module work;
- `projects/of3_all_atom/model.py`, end of `run_trunk`: `return s_input.float(), s.float(), z.float()`,
  unconditional.

Four forward-only arms, same batch by hash, same draws replayed at 0 mismatches, same pinned RNG
state, only the patching different (`d19_forward_discriminator.py`):

| arm | autocast | `Tensor.float()` | loss |
|---|---|---|---|
| A | neutralised | identity on float64 | 1.6422035029890711 |
| B | neutralised | left alone | `RuntimeError: expected scalar type Float but found Double` |
| C | **left alone** | identity on float64 | **1.6422035029890711** |
| D | left alone | left alone | `RuntimeError: expected scalar type Float but found Double` |

**A and C are bit-identical.** Their fp32 autocast blocks contribute exactly nothing on this path,
so autocast is eliminated as a candidate — and half of the reference's `no_autocast` context is
dead weight. Only the `Tensor.float` patch is load-bearing, which is worth stating plainly: the
reference is their function with one unconditional downcast removed, and nothing else.

**B and D do not produce a number, they crash.** A float64 replay that leaves `run_trunk`'s
`.float()` in place cannot reach the loss at all: the first LayerNorm downstream gets float32
activations against float64 weights. So the trunk downcast cannot silently account for 1.6311
either.

That leaves the question sharper than it was. There are only two ways past that `RuntimeError`:
neutralise the `.float()` on float64 tensors, which is what the reference does and which gives
1.6422035029890711, or let the trunk output downcast and move the diffusion conditioning to
float32 to match it — a genuinely lower-precision function. **How the qb2 side got past
`run_trunk`'s unconditional `.float()` is now the open question**, and it is the one thing left
that would move a float64 loss by 1e-2 with the draws, the batch and the weights all pinned.

Handed to `of3t-gradients`, who owns that side.

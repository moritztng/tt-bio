# Changing a model and running it on Tenstorrent

There is no compiler today that takes an arbitrary changed PyTorch model and runs it well on a
Tenstorrent chip. What tt-bio has instead: every model it ships is a PyTorch model whose heavy
layers run as hand-written ttnn code, and most changes people want to make to a model they
already use (a new output head, a different loss, a retrained head) sit outside those layers and
run as ordinary PyTorch next to them. This page says which change takes which route, what each
costs, and where each stops.

## Which change, which route

| You want to | Route | Runs where | What you write |
|---|---|---|---|
| Read a new output off a fold (a contact head, a probe, a score) | `tt-bio predict --head` | trunk on the chip, your head on the host | one Python file |
| Train that head with your own loss | plain PyTorch on what the head saw | host | one training script |
| Add or reweight a BindCraft 2 loss term | the BindCraft 2 custom loss ([`bindcraft2.md`](bindcraft2.md)) | chip, through tt-bio's own autograd | a Python function |
| Fine-tune the model's own weights on your objective | `tt_bio.train` ([`training.md`](training.md)) | chip | an objective, or your own loop |
| Change a layer inside the trunk | a ttnn implementation of that layer, checked against yours | chip | a port of that module, see [`porting-a-model.md`](porting-a-model.md) |
| Run a model tt-bio does not ship | a port | chip | see [`porting-a-model.md`](porting-a-model.md) |

The first two rows need no Tenstorrent-specific code at all. The last two are real engineering:
a changed layer is new device code plus a parity check, and that is the work the porting guide
describes.

## Custom output heads: `--head`

A head is a function, or an `nn.Module` class, in your own file. tt-bio calls it once per
prediction with what the fold computed and writes whatever it returns next to the structure:

```bash
tt-bio predict target.yaml --model boltz2 --head my_heads.py:ContactHead
# boltz2_results_target/structures/target.cif
# boltz2_results_target/structures/target_ContactHead.npz
```

```python
import torch

class ContactHead(torch.nn.Module):          # a class is built once per worker, no arguments
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(128, 1)
        self.load_state_dict(torch.load("contact_head.pt"))

    def forward(self, fold):                  # fold: tt_bio.heads.Fold
        return {"contact_probs": torch.sigmoid(self.proj(fold.z).squeeze(-1))}
```

What `fold` holds, with the padding a batch adds already removed, so the token axis lines up
with the structure file:

| field | Boltz-2 | ESMFold-2 | |
|---|---|---|---|
| `s` | `[N, 384]` | `[N, 451]` | Boltz-2: single representation after the last trunk pass. ESMFold-2 has no single track in its trunk, so this is its input embedding |
| `z` | `[N, N, 128]` | `[N, N, 256]` | pair representation after the last trunk pass |
| `coords` | `[samples, atoms, 3]` | same | every diffusion sample, Angstrom |
| `plddt` | `[samples, N]` | same | 0 to 1 |
| `pae`, `pde` | `[samples, N, N]` | same | Angstrom |
| `pred`, `feats` | dicts | dicts | the raw prediction and input batch, padded, for anything else |

Repeat `--head` for several heads. Each writes `<record>_<NAME>.npz`.

**A head cannot change the fold.** It runs after the structure and the confidence values have
been written, so a fold with a head writes the same structure as one without it, plus the npz.
With no `--head` none of this code runs.

**Boltz-2 and ESMFold-2 (`esmfold2`, `esmfold2-fast`) for now.** Their folds return the trunk
representations to the host, so a head reads them without extra device traffic. Boltz-2 runs on
CPU, GPU and Tenstorrent; ESMFold-2 runs on Tenstorrent only, and its first fold downloads the
25.4 GB ESMC-6B language model it is built on, 21 minutes at 20 MB/s. The two models' `z` differ in width,
so a head trained on one does not load on the other. `predict` refuses `--head` for any other model
rather than ignoring it.

**Your code runs where you run tt-bio.** `--head` loads your file in the local worker. It is
refused with `--controller`, and JapanFold does not run customer code.

## A worked example: a contact head with its own loss

[`examples/custom_head/`](../examples/custom_head/) takes a real modification from source to a
running fold: it exports Boltz-2's pair representation for six small proteins, trains a contact
head on it with a custom loss that weights long-range pairs four times, and then folds a protein
it never saw with that head attached. Its README lists every command and the output of a run.

The same commands run on a laptop with `--accelerator cpu --no_kernels` and on a Tenstorrent card
without either flag. Only the trunk moves; the head and its training are plain PyTorch in both.

## Where each route stops

- **A head reads the trunk, it does not train it.** The gradient of a head's loss stops at `z`.
  To move the trunk's own weights you need `tt_bio.train`, whose backward runs through tt-bio's
  hand-written autograd, and today a dataset and featuriser are wired up for OpenFold3 only
  ([`training.md`](training.md)).
- **A changed layer is a port.** Each trunk module is ttnn code written to match its reference.
  A different triangle update or attention variant needs its own implementation and the same
  component-by-component parity check every shipped model went through.
- **OpenFold3, Protenix, OpenDDE, RF3 and AF2-IG have no `--head` yet.** Their folds keep the
  trunk on the chip or return a different result object; each needs its own small adapter.

## The compiler route

Tenstorrent's compiler stack is [tt-xla](https://github.com/tenstorrent/tt-xla) (PyTorch through
torch-xla, JAX through `jax.jit`) on top of tt-mlir. As of October 2026 it is an alpha
(`Development Status :: 3 - Alpha` in its `setup.py`), ships as nightly wheels on Tenstorrent's
own package index rather than PyPI, and needs Python 3.12. Its own model CI lists Boltz-2 as not
supported, "Hangs or takes forever to run"
([test config](https://github.com/tenstorrent/tt-xla/blob/main/tests/runner/test_config/torch/test_config_inference_single_device.yaml)),
and has no entry for ESM, ESMFold, Protenix, OpenFold or AlphaFold. tt-bio does not use it, and
we have no accuracy or speed measurement of a structure model through it. It is the right thing to
try for a small, conventional network; it is not yet a way to run a modified folding model.

## Running on your own hardware

Everything on this page runs on hardware you hold. `pip install 'tt-bio[tenstorrent]'` on a host
with a Tenstorrent Wormhole or Blackhole card, or plain `pip install tt-bio` for the Boltz-2 CPU
and GPU path, which is enough to develop and test a head before it ever sees a chip. `--head`
needs tt-bio 0.13.0 or later.

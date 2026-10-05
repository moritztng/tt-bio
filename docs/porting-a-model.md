# Porting a new model to tt-bio

This is how a model gets into tt-bio, what it took for the ports already in it, and what we need
from you if you would rather we did it. For a change to a model tt-bio already runs, start with
[`extending.md`](extending.md): most changes do not need a port.

## What a port is

A tt-bio model is the upstream reference with its neural network re-implemented in ttnn, the
Tenstorrent op library. The reference's own host code (parsing, featurisation, writing the
structure) is kept and vendored under `tt_bio/_vendor/<model>` with its licence. Only the learnable
modules are rewritten, each one checked against the reference module it replaces. There is no
automatic translation step: the compiler route is described, with its current status, in
[`extending.md`](extending.md#the-compiler-route).

## How long it took

Calendar time from the repository's history, by people and agents who already knew the codebase
and had the shared ttnn building blocks (attention, triangle updates, transitions) to reuse.
Upstream reading before the first commit is not counted.

| model | first commit | whole model on the chip | parity signed off | commits | ttnn code written | reference code vendored |
|---|---|---|---|---|---|---|
| SaProt (650M, 35M) | 2026-07-17 | same day | same day | 20 | 831 lines | none, the reference is a pip package |
| AF2-IG | 2026-08-20 | not recorded | 2026-08-23 | 52 | 2,791 lines plus a 1,333-line torch reference | none, the reference is that translation |
| ESMFold-2 | 2026-06-02 | 2026-06-03 | 2026-07-09 | 112 | 2,523 lines | 16,510 lines |
| OpenFold3 | 2026-07-12 | 2026-07-13 | 2026-08-07 | 174 | 4,679 lines | 35,881 lines |

The pattern is the same in every row: getting each component to match the reference takes days.
The weeks after that go to whole-model bugs that only show on real targets, and to sizes,
precision modes and speed. A model that reuses an existing family (SaProt on the ESM-C port)
is a day; an AlphaFold 3-style model with its own conventions is about a month to sign-off.

## The steps

1. **Pin the reference.** Read its real inference defaults (recycles, sampling steps, samples)
   from its code, not its paper. Vendor its host code under `tt_bio/_vendor/<model>`, depending
   only on pip wheels, and read the architecture off the checkpoint's keys.
2. **Build a CPU float32 golden harness** that runs the reference and saves every module's
   inputs and outputs for a few targets.
3. **Map the weights** losslessly. Every array in the checkpoint is either loaded or accounted
   for by name.
4. **Port one module at a time**, smallest model variant first, each one matched against its
   golden output before the next starts: PCC above 0.98, and a relative L2 bound as well (below).
5. **Run the whole model on the chip** with real weights, the host doing glue only, and check
   the structure against ground truth.
6. **Parity against the reference**: device against reference structures, judged against the
   reference's own seed-to-seed spread on the same targets.
7. **Sizes**: every size up to 1,536 tokens folds, or the measured ceiling is recorded and enforced.
8. **Speed**, measured warm with the clock recorded: load time, then fold time, then the
   block-fp8 `--fast` mode.
9. **Unify**: the shared CLI, worker, output layout and docs, so the new model is one more
   `--model` value rather than a separate tool.

[`model-bringup-checklist.md`](model-bringup-checklist.md) is the checklist each port is held to.

## The accuracy bar

- **Per module:** PCC above 0.98 against the reference module, plus a relative L2 bound. PCC
  alone cannot see a scale error.
- **Whole model:** the device's structures sit inside the reference's own seed-to-seed spread on
  the same targets. Bit-exactness is not required: bf16 hardware arithmetic and a float32 CPU
  reference never agree to the last bit, and a seed change moves a structure more than that.
- **An optimisation that claims to change nothing** is proved with a bit-identical intermediate,
  not with a fold RMSD, because two folds of the same code already differ.

## The traps that cost the most time

Each of these passed every module-level check and was only found on whole targets.

- **Two inputs of the same total width, swapped.** Every shape check passes; the structure is
  7 to 9 A off. Test each module with inputs that are not interchangeable.
- **A shared kernel built for one model's convention.** A bias scaled one family's way, used in
  another's, gave a 12.6 A cross-chain error while the pair representation still read 0.977 PCC.
  Check every reused block against the new reference, not the old one.
- **A scale error at near-perfect PCC.** A distogram symmetrised twice read 0.9998 PCC with a
  maximum error of 6.8. That is why the relative L2 bound exists.
- **Fused attention on near-degenerate logits.** It flattens them, and on one target sent the
  single representation to 0.44 PCC. Some attention sites need an explicit float32 softmax for
  correctness, not speed.
- **What a feature diff cannot see.** A stereochemistry step left out of vendoring, an alignment
  cap that silently did not apply, an RDKit version that moves reference conformers. Diff the
  featurised tensors against the reference's, not just their keys.
- **bf16 drift that compounds across recycles.** Small per-step errors add coherently over a
  recycling loop, so confidence can drift at long lengths even when each step matches.

## If you would rather we port it

For a model you cannot port yourself, the offer is that we port it into tt-bio, where it then
ships to everyone who installs it. That is a conversation with us, and it needs four things from
you:

1. **The weights**, or where to get them, and **their licence**: it must allow us to redistribute
   or download them for users, and to run them on our own hardware if you also want it on
   JapanFold. Weights are often licensed separately from the code.
2. **A reference implementation** that runs: the code you trust, at the commit you trust, with the
   inference settings you use.
3. **A test case**: a few inputs with the outputs the reference gives on them, ideally with
   ground-truth structures, so parity is checked against what you consider correct.
4. **What matters to you**: the sizes you fold, the outputs you read, and whether speed or
   exactness comes first.

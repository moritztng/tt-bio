# ABodyBuilder3 port: what each gate proves

Each script answers one question. They are ordered by what they rest on: a number from a
lower row is only meaningful if the rows above it pass. `train-b3-train` owns the schedule and the
run; this directory is how it can re-establish that the port is still what these numbers say.

All of them take `PYTHONPATH=$PWD`. The device ones need `TT_VISIBLE_DEVICES=<card>` and
`TT_BIO_LEASE_CARDS=<card>`; the host ones need neither and run in CI.

| script | question | last result | needs |
|---|---|---|---|
| `reference_gate.py` | is `abodybuilder3_reference.py` upstream's model? | tables 0.0, IPA 1.847e-13, worst model output 7.3e-12 on `unnormalized_angles` against a 1e-10 bar in float64 | upstream checkout |
| `../../tests/test_abodybuilder3_reference.py` | has the reference moved since? | 4 pinned digests, negative control breaks it | nothing |
| `precision_probe.py` | what precision does the card give? | matmul 1.25e-03, eltwise 3.0e-07, reductions ~1e-03 | card |
| `relayout_probe.py` | which reorientations are exact? | leading-axis permute and reshape exact, last-axis 7.5e-04 | card |
| `op_gradcheck.py` | is every op's gradient right? | 27 cases, forward and backward vs torch float64, grad-off vs grad-on 0.00e+00 | card |
| `device_gate.py` | is the IPA block right? | forward 6.74e-03, gradients 1.3e-03..1.8e-02 | card |
| `model_gate.py` | is the 8-block model right? | `perf/abb3_port/model_gate_qb1c1.txt`, and one row is over its bar | card |
| `loss_gate.py` | are the losses upstream's? | all terms 0.00e+00 vs their `loss.py` in float64 | upstream checkout |
| `output_gate.py` | does the PDB writer round-trip? | 52 structures, worst region 0.0000 A | Zenodo `output/` |
| `fold_gate.py` | does the port reproduce their predictions? | reference exact on 17, device 0.020 A mean CDR-H3 | Zenodo `output/` + card |
| `moment_audit.py` | did a run ever train its parameters? | `base-loss` step 1,248: 80 live, 356 dead of 436, 2,704 of 7,992,080 scalars (0.034 %) under a live `exp_avg` | a checkpoint |
| `../../tests/test_gradient_reaches_every_parameter.py` | does one real step reach every parameter? | red on today's tree: 380 of 436 take an identically zero gradient from the run's own starting weights, control arm green | card |
| `step_time.py` | what does the device half of a step cost? | 14.472 s median over 100 steps | card |
| `step_gate.py` | what does a COMPLETE step cost? | `perf/abb3_port/step_gate_default_qb1c1.txt` at the shipped default, and `step_gate_complete_qb1c3.txt` for the attribution | card |

## The two that decide whether a number is real

**`fold_gate.py`** is the one that matters most and the cheapest to misread. It scores through
`tt_bio/antibody_rmsd.py`, which is `train-b1-instrument`'s and is validated by recomputing
Kenlay et al.'s own released per-structure values. Do not write another scorer: the published
"backbone RMSD" is over **N, CA, C, CB**, and our atom14 layout's first four slots are
**N, CA, C, O**, so slicing `[..., :4]` off our own output scores a different quantity and lands
about 0.07 A off, a third of the 0.20 A accuracy bar and invisible to it.

**`step_gate.py`** asserts two properties before it reports a time, and both were violated when it
was first written: the padded weight channels must take zero gradient, and they must still be zero
after an optimizer step. That is what lets RAdam update the device-layout parameters in place. If
either assertion fires, the port has grown a channel whose structural zero is not protected on the
backward, and the timing is the least of the problems.

## A recorded number says which commit it came from

`model_gate_qb1c3.txt` was recorded at `ca7a7573e` and six later commits on the same branch moved
the model it scores. Nothing noticed. The file went on reading 0.014 A where the head reads
0.018 A, and a difference that size reads exactly like a difference between two card types: it was
nearly filed as p150a against p300c before two cards agreed and the file turned out to be the
outlier.

So an artifact here carries the commit it was recorded at:

    RECORDED-AT: <commit> <the script that produced it>

`tests/test_recorded_claims.py` fails when that code differs from what the number was recorded
against, and names the files. It reads content rather than the commit log, which is what keeps a
merge that brings the recording's own branch in from being reported as the cause. It does not read
a hand-written list of the code either, because a hand-written list goes stale the same silent
way: it takes the producing script and follows its module-level imports. Re-run the script and
update both the numbers and the line. Deleting the artifact is not the fix, and the test says so
by name.

Expect a red on a work branch. The code this closure reaches takes about 19 commits a day that
change executable source, so a recording goes unconfirmed within a median of 15 minutes of being
made. That is the
shelf life of a device number here, not a fault in the check, and the affordable place to
re-record is the commit you release from.

The same file keeps each script's `Run:` line equal to its own argparse defaults. `step_gate.py`
shipped `--micro 8` in both and it OOMs in the first backward, so the one configuration the docs
sent a reader to was the one no recorded run had ever used.

## What a reproduction still needs from outside this directory

* `data.tar.gz` staged (3.0 GB, on qb2 and qb1). Their `structures/*.pt` carry every loss target,
  so no featuriser is owed beyond the two input one-hots, and those are built on the card by
  `tt_bio/train/abb3_features_device.py` from the `(micro, n_tok)` index maps rather than
  uploaded: the pair map is 138 MB a micro-batch and the index it expands is 8 KB.
  `single_and_pair_features` stays the host reference that path is checked against.
* The three violation terms, which upstream gates on `finetune` and stage 1 therefore never calls.
* A decision on their supervised-chi pi-periodicity defect, which `loss_gate.py` prints the size of
  on every run. Ours reproduces it by default because the target is their number.

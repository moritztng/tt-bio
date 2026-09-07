# Bringing up a new model

A model is not "brought up" until every box below is ticked. Running is the first step, not the
last one.

## 1. It runs, and it is right

- [ ] `tt-bio predict --model <m> examples/prot.yaml --override` completes, the output CIF parses,
      and its Kabsch RMSD against ground truth is in the expected band. Runs without error, correct
      output, and accurate structure are three different claims. Check all three.
- [ ] A parity leg exists in `scripts/full_parity_gate.py` with a cached reference, and it passes.
- [ ] A perf cell exists in `scripts/perf_regression.py`.

## 2. It fits at the size you advertise

- [ ] `scripts/capacity_gate.py` passes for the model at the bar (1504 tokens), at the MSA depth
      it is actually served with:

      TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
        python3 scripts/capacity_gate.py --models <m>

      A screen-only pass is not a pass. The gate reports `INCONCLUSIVE` for a Tier 1 run, because
      one block cannot see the cumulative-residency failures.
- [ ] If it fails the bar, `tt_bio/size_limits.CEILINGS` carries a row for this architecture with
      the measured `pass_at`, the `fail_at` negative control, and evidence naming the gate run. A
      model that refuses above a measured ceiling is shipped. A model that crashes there is not.
- [ ] `docs/capacity_gate_baseline.json` is re-recorded, so `tests/test_capacity_gate.py` pins the
      ceiling table the gate was measured against.

The capacity gate answers "does it allocate and complete". It cannot answer "is the output right",
and it is meant to run on cards that miscompute. It does not substitute for the parity gate, and
neither substitutes for the other.

## 3. Nothing hand-lists it

The model must appear automatically in every gate. Add it to the right tuple in `tt_bio/main.py`
(`PREDICT_MODELS` via `_MODEL_RESULTS_PREFIX`, `EMBED_MODELS`, `DESIGN_MODELS`, ...) and the
derived rosters follow. Then run:

    PYTHONPATH="$PWD" python3 -m pytest tests/test_size_limits.py tests/test_capacity_gate.py -q

Both suites fail loudly on a model that is shipped but uncovered. If one tells you to write an
exemption, write a reason, not a placeholder: an exemption records what is not covered so the gap
stays readable.

## 4. The docs match reality

- [ ] `README.md` names the model and does not promise a size it throws at.
- [ ] `CHANGELOG.md` has an entry.
- [ ] Any published ceiling matches the measured one.

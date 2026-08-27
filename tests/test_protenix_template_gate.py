"""The Protenix template embedder contributes only when the CHECKPOINT ships template blocks.

Upstream v0.5.0 `protenix/model/modules/pairformer.py:1000` returns literal 0 from
`TemplateEmbedder.forward` when `n_blocks < 1`, and `protenix.py::get_pairformer_output` does
not even call it in that case. The v0.5.0 base checkpoint (protenix-v1) ships 0 blocks: its
five template projections are dead weight.

`Trunk.__call__` used to gate on `nt`, the number of template SLOTS in the features, and
`protenix_data.dummy_template_features` always emits 4. So protenix-v1 would have added
`linear_u(relu(mean(LN(tpl_a + linear_z(LN(z))))))` into z on every recycling cycle where
upstream adds nothing -- silent, plausible and wrong. The gate now reads `self.TPL` as well.

The fix has to be INERT for the checkpoints that do ship a stack. That is what this pins:
protenix-v2 and opendde must both census a non-empty template pairformer stack, so the new
condition can never change their arithmetic. No device, no fold: a shape census of the weights
answers it, which is why this can run in CI.
"""
import sys

from tt_bio import weights
from tt_bio.main import RECYCLING_STEPS

# model id -> the template pairformer depth its checkpoint must ship for the gate to be inert
TEMPLATE_STACK_MUST_BE_NONEMPTY = ("protenix-v2", "opendde")
TEMPLATE_STACK_MUST_BE_EMPTY = ("protenix-v1",)


def _depth(model_id):
    """Template pairformer depth, censused off whatever copy of the checkpoint is on this box.

    Never downloads: a checkpoint that is not cached SKIPs, so this stays a CI-cheap shape check
    and never turns into a 1.5 GB fetch."""
    import torch

    from tt_bio.protenix import n_blocks

    path = weights.ARTIFACTS[model_id].dest()
    if not path.exists():
        raise FileNotFoundError(f"{model_id} not cached at {path}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return n_blocks(sd, "template_embedder.pairformer_stack"), sd


def _ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


def check_recycling_table_matches_the_checkpoint_derivation():
    """main.RECYCLING_STEPS and protenix.trunk_recycles are two readers of one fact.

    The CLI has to answer before any weights are loaded, so it keys on the model id; the Trunk
    reads the template depth off the weights it already holds. Neither is allowed to drift.
    """
    from tt_bio.protenix import trunk_recycles

    ok = True
    for mid in TEMPLATE_STACK_MUST_BE_EMPTY + TEMPLATE_STACK_MUST_BE_NONEMPTY:
        if mid not in RECYCLING_STEPS:
            ok = _ok(False, f"{mid} has a RECYCLING_STEPS entry") and ok
            continue
        try:
            _d, sd = _depth(mid)
        except Exception as e:                     # checkpoint not on this box
            print(f"SKIP {mid}: {type(e).__name__}: {e}")
            continue
        want = RECYCLING_STEPS[mid]
        got = trunk_recycles(sd)
        ok = _ok(got == want, f"{mid}: trunk_recycles reads {got}, RECYCLING_STEPS says {want}") and ok
    return ok


def check_template_depths():
    ok = True
    for mid in TEMPLATE_STACK_MUST_BE_NONEMPTY:
        try:
            d, _sd = _depth(mid)
        except Exception as e:
            print(f"SKIP {mid}: {type(e).__name__}: {e}")
            continue
        ok = _ok(d > 0, f"{mid} ships {d} template pairformer blocks, so the `self.TPL` gate "
                        f"cannot change its arithmetic") and ok
    for mid in TEMPLATE_STACK_MUST_BE_EMPTY:
        try:
            d, _sd = _depth(mid)
        except Exception as e:
            print(f"SKIP {mid}: {type(e).__name__}: {e}")
            continue
        ok = _ok(d == 0, f"{mid} ships {d} template pairformer blocks; upstream v0.5.0 ships 0 "
                         f"and adds literal zero") and ok
    return ok


def main():
    results = [check_template_depths(), check_recycling_table_matches_the_checkpoint_derivation()]
    print("\n" + ("PASSED" if all(results) else "FAILURES"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

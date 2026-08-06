"""Per-model default for REQUESTED diffusion sampling steps (--sampling_steps unset).

esmfold2/esmfold2-fast request 100, the ESMFold2 paper's benchmark protocol (A.2.11:
"We use N = 100 which reduces to 68 sampling steps" — the Karras schedule is clipped at
sigma_max=256, so a request of 100 executes 68 denoise steps). Every other model keeps 200.
An explicit --sampling_steps is honored verbatim for every model. Host-only — no device,
no network.
"""
from __future__ import annotations

import pytest

from tt_bio.main import _resolve_sampling_steps as _resolve


@pytest.mark.parametrize("model", ["esmfold2", "esmfold2-fast"])
def test_esmfold2_default_requests_100(model):
    """Unset -> esmfold2/esmfold2-fast request 100 (= 68 executed after the sigma-clip)."""
    assert _resolve(None, model) == 100


@pytest.mark.parametrize("model", ["boltz2", "protenix-v2", "opendde", "opendde-abag"])
def test_other_models_keep_200(model):
    """Unset -> every non-esmfold2 model keeps the historical 200 (unchanged behavior)."""
    assert _resolve(None, model) == 200


@pytest.mark.parametrize("model", ["boltz2", "esmfold2", "esmfold2-fast"])
@pytest.mark.parametrize("n", [14, 50, 68, 100, 200])
def test_explicit_value_overrides_for_every_model(model, n):
    """An explicit --sampling_steps is honored verbatim, regardless of model — including
    68 (requested != executed is the caller's choice to make, not the resolver's)."""
    assert _resolve(n, model) == n

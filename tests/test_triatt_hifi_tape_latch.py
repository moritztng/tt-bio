"""A taped decline on the fused-HiFi triangle-attention route must not retire its config.

`triatt_sdpa.sdpa` returns None under an open tape because `generic_op` has no backward. The
ladder in `_tri_att_sdpa_hifi_inner` used to read that None as the config failing and add it to
`_TRIATT_HIFI_OVER_L1`, so one taped forward switched the route off for every untaped forward
later in the same process. BindCraft 2 alternates taped and untaped passes, and bcx-forward
measured 208 of 208 untaped calls declined after the first taped one (perf/bcx_forward/
latch_before_n256.json on wk/bcx-forward).

Untaped, a decline still retires the config exactly as it did before, so no inference call
changes route. Device-free: the kernel is a stub.
"""
import types

import pytest

T = pytest.importorskip("tt_bio.tenstorrent")


class _Q:
    shape = (1, 4, 256, 32)
    dtype = "bf16"


@pytest.fixture
def ladder(monkeypatch):
    state = {"taping": False, "declines_untaped": False, "calls": 0}

    def sdpa(q, k, v, bias, scale, q_chunk, k_chunk, **kw):
        state["calls"] += 1
        if state["taping"] or state["declines_untaped"]:
            return None
        return "served"

    monkeypatch.setattr(T, "_TRIATT_HIFI_OVER_L1", set())
    monkeypatch.setattr(T, "_tri_att_fused_large_s", lambda *a, **kw: None)
    monkeypatch.setattr(T, "_triatt_sdpa", types.SimpleNamespace(sdpa=sdpa))
    monkeypatch.setattr(T.ops, "taping", lambda: state["taping"])
    return state


def _call():
    q = _Q()
    return T._tri_att_sdpa_hifi_inner(q, q, q, None, 1.0)


def test_taped_decline_leaves_the_route_open_for_untaped_calls(ladder):
    ladder["taping"] = True
    assert _call() is None
    assert T._TRIATT_HIFI_OVER_L1 == set()
    ladder["taping"] = False
    assert _call() == "served"


def test_untaped_decline_still_retires_the_config(ladder):
    ladder["declines_untaped"] = True
    assert _call() is None
    retired = set(T._TRIATT_HIFI_OVER_L1)
    assert retired, "an untaped decline retires its config, as on main"
    calls = ladder["calls"]
    ladder["declines_untaped"] = False
    assert _call() is None, "a retired config is not retried"
    assert ladder["calls"] == calls

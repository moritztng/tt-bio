#!/usr/bin/env python3
"""Lever 2 must not change what ESMFold2 gets. Verified without a device.

ESMCHiddenStatesModel is shared: ESMFold2's LanguageModelShim consumes all n_layers+1 hidden
states, and the embed path uses only the last. The new last_hidden_only flag defaults to False,
but "it defaults to False" is a claim about a default, not about behaviour. This exercises the
real __call__ with a stubbed self and stubbed ttnn, and counts the _to_host calls each way.

Stubbing rather than loading 12.6 GB of weights is deliberate: the branch under test is pure
control flow over the readback, and a device run of the 6B is queued behind a saturated lock.
"""
import json
import sys
import types

from tt_bio import esmc as E

out = {}


def build_fake(n_layers):
    calls = []

    class FakeT:
        def __init__(self, tag):
            self.tag = tag
            self.shape = [1, 8, 16]

    self = types.SimpleNamespace()
    self.n_heads = 2
    self.n_layers = n_layers
    self.device = None
    self.compute_kernel_config = None
    self.norm_weight = FakeT("norm_w")
    self.norm_weight.shape = [16]
    self.embed = lambda tokens: FakeT("embed_out")
    self.blocks = [(lambda x, c, s, m, k, i=i: FakeT(f"block{i}")) for i in range(n_layers)]

    def to_host(t):
        calls.append(t.tag)
        return f"host({t.tag})"

    self._to_host = to_host
    return self, calls


def run(n_layers, last_hidden_only):
    self, calls = build_fake(n_layers)
    orig_rope, orig_ln, orig_dealloc = E.rope_tables, E.ttnn.layer_norm, E.ttnn.deallocate
    E.rope_tables = lambda *a, **k: (None, None)
    E.ttnn.layer_norm = lambda x, **k: types.SimpleNamespace(tag="norm_out")
    E.ttnn.deallocate = lambda x: None
    try:
        tokens = types.SimpleNamespace(shape=[1, 8])
        hidden = E.ESMCHiddenStatesModel.__call__(
            self, tokens, None, None, last_hidden_only=last_hidden_only)
    finally:
        E.rope_tables, E.ttnn.layer_norm, E.ttnn.deallocate = orig_rope, orig_ln, orig_dealloc
    return hidden, calls


N = 80  # esmc-6b
hidden_default, calls_default = run(N, False)
hidden_flag, calls_flag = run(N, True)

out["default_n_states"] = len(hidden_default)
out["default_expected"] = N + 1
out["default_readbacks"] = len(calls_default)
out["default_unchanged"] = len(hidden_default) == N + 1 and len(calls_default) == N + 1

out["flag_n_states"] = len(hidden_flag)
out["flag_readbacks"] = len(calls_flag)
out["flag_only_last"] = len(hidden_flag) == 1 and len(calls_flag) == 1
out["readbacks_saved"] = len(calls_default) - len(calls_flag)

# the last element must be the SAME state either way -- that is the whole contract
# _trunk_forward's [-1] depends on
out["default_last"] = hidden_default[-1]
out["flag_last"] = hidden_flag[-1]
out["last_state_identical"] = hidden_default[-1] == hidden_flag[-1] == "host(norm_out)"

# the default must still be False at both signatures ESMFold2 can reach
import inspect
for fn, name in ((E.ESMCHiddenStatesModel.__call__, "ESMCHiddenStatesModel.__call__"),
                 (E.ESMCLanguageModel.forward, "ESMCLanguageModel.forward")):
    sig = inspect.signature(fn)
    out[f"default_false_{name}"] = sig.parameters["last_hidden_only"].default is False

# and ESMFold2's own entry point must not mention it at all
src = inspect.getsource(E)
out["trunk_forward_opts_in"] = "last_hidden_only=True" in inspect.getsource(E._trunk_forward)
out["opt_in_sites"] = src.count("last_hidden_only=True")

checks = [out["default_unchanged"], out["flag_only_last"], out["last_state_identical"],
          out["readbacks_saved"] == N,
          out["default_false_ESMCHiddenStatesModel.__call__"],
          out["default_false_ESMCLanguageModel.forward"],
          out["trunk_forward_opts_in"], out["opt_in_sites"] == 1]
out["ALL_PASS"] = all(checks)
print(json.dumps(out, indent=2))
sys.exit(0 if out["ALL_PASS"] else 1)

"""RF3's confidence heads can be reduced one sample at a time, host-only.

Stacked, the heads keep every sample's [I, I, 64] PAE and PDE logits in fp32 on the host:
0.6 GB a sample at 1088 tokens, so `--diffusion_samples 100` there died on the host OOM
killer after its rollout had finished. The worker now passes a `per_sample` reducer, and
these tests pin that it sees exactly what the stacked path would have indexed, and that the
worker's rf3 path uses it.

The real `RF3.confidence` loop runs with the device calls stubbed.
"""
from __future__ import annotations

import inspect
import types

import torch

from tt_bio.rf3 import model as rf3_model
from tt_bio.rf3.model import RF3
from tt_bio.worker import _WorkerState


def _stubbed(monkeypatch, n_tok=7):
    monkeypatch.setattr(rf3_model, "distance_onehot", lambda x, idx, dev: x)
    monkeypatch.setattr(rf3_model.ttnn, "to_torch", lambda t: t)

    def head(s_inputs, s, z, x):
        # A head output that depends on the member's own coordinates, as the real one does.
        k = float(x.sum())
        return {"pae_logits": torch.full((1, n_tok, n_tok, 64), k),
                "plddt_logits": torch.full((1, n_tok, 50), -k)}
    return types.SimpleNamespace(confidence_head=head, device=None)


def test_per_sample_sees_what_the_stacked_path_indexes(monkeypatch):
    stub = _stubbed(monkeypatch)
    x_pred = torch.randn(4, 11, 3)
    stacked = RF3.confidence(stub, None, None, None, x_pred, None)
    seen = []

    def reduce(d, heads, x):
        seen.append(d)
        assert torch.equal(x, x_pred[d])
        for k, v in heads.items():
            assert torch.equal(v[0], stacked[k][d]), k
        return {"d": d, "mean_pae": float(heads["pae_logits"].mean())}

    got = RF3.confidence(stub, None, None, None, x_pred, None, reduce)
    assert seen == [0, 1, 2, 3]
    assert [r["d"] for r in got["per_sample"]] == [0, 1, 2, 3]
    assert set(got) == {"per_sample"}, "no stacked logits may ride along with the reductions"


def test_predict_forwards_the_reducer_and_the_worker_passes_one():
    assert "per_sample" in inspect.signature(RF3.predict).parameters
    assert "per_sample" in inspect.getsource(RF3.predict).split('"""')[-1]
    src = inspect.getsource(_WorkerState._predict_rf3_one)
    assert "per_sample=" in src and 'got["per_sample"]' in src, (
        "the rf3 fold path must reduce each sample as it lands, not stack every sample's "
        "logits and index them afterwards")

"""Parity tests for the PLAPT standalone affinity head (tt_bio.affinity) vs the
from-scratch PyTorch reference in tests/affinity_reference.py.

Component-by-component (the tt-bio idiom): identical weights into both, run,
compare PCC. Random-weight parity first (architecture correctness), then
real-weight parity on captured I/O. Runs on TT device 1 (one device context).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from affinity_reference import (  # noqa: E402
    ChemBERTaReference,
    FusionHead as RefFusionHead,
    make_chemberta,
    load_chemberta_config,
)
from tt_bio.tenstorrent import WeightScope, get_device  # noqa: E402
from tt_bio import affinity as tt_aff  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(893)

CFG = load_chemberta_config()


def pcc(a, b):
    a = torch.as_tensor(a).flatten().float()
    b = torch.as_tensor(b).flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ref_to_tt_keys(sd: dict) -> dict:
    """ChemBERTaReference state_dict -> tt_bio.affinity key layout."""
    import collections
    out = collections.OrderedDict()
    for k, v in sd.items():
        if k.startswith("encoder."):
            out[k.replace("encoder.", "layer.", 1)] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Fusion head (real ONNX-extracted weights)
# ---------------------------------------------------------------------------

def test_fusion_head_real_weights():
    sd = tt_aff._fusion_head_state_dict()  # prot_w, mol_w, bn_*, l1/l2/fl ...
    ref = RefFusionHead().eval()
    ref.load_state_dict(sd, strict=False)
    tt = tt_aff.AffinityHead.from_pretrained()

    prot = torch.randn(4, 1024)
    mol = torch.randn(4, 768)
    ref_out = ref(prot, mol)
    tt_out = tt(prot, mol)
    print("fusion PCC", pcc(ref_out, tt_out), "maxdiff",
          (ref_out - tt_out).abs().max().item())
    assert pcc(ref_out, tt_out) > 0.98
    # rescale sanity
    aff = tt_aff.AffinityHead.to_affinity(tt_out)
    assert aff.shape == (4, 1)


# ---------------------------------------------------------------------------
# ChemBERTa ligand encoder (random-weight component parity)
# ---------------------------------------------------------------------------

def _make_pair(seq_len=64, seed=0):
    ref = make_chemberta(seed)
    sd = _ref_to_tt_keys(ref.state_dict())
    tt = tt_aff.ChemBERTa(CFG)
    tt.load_state_dict(sd, strict=False)
    ids = torch.randint(5, CFG["vocab_size"], (2, seq_len))
    return ref, tt, ids


def test_chemberta_embeddings(seq_len=64):
    ref, tt, ids = _make_pair(seq_len, seed=1)
    ref_x = ref.embeddings(ids)
    pos = tt_aff._position_ids(ids, CFG["pad_token_id"])
    tokens_tt = _to_tt_ids(tt, ids)
    pos_tt = _to_tt_ids(tt, pos)
    tt_x = tt.module.embed(tokens_tt, pos_tt)
    tt_x = torch.Tensor(ttnn_to_torch(tt_x)).float()
    print("embed PCC", pcc(ref_x, tt_x))
    assert pcc(ref_x, tt_x) > 0.98


def test_chemberta_layer(seq_len=64):
    ref, tt, ids = _make_pair(seq_len, seed=2)
    ref_x = ref.embeddings(ids)
    ref_y = ref.encoder[0](ref_x)
    # feed the SAME post-embedding tensor into the ttnn layer
    pos = tt_aff._position_ids(ids, CFG["pad_token_id"])
    tokens_tt = _to_tt_ids(tt, ids)
    pos_tt = _to_tt_ids(tt, pos)
    tt_x = tt.module.embed(tokens_tt, pos_tt)
    tt_y = tt.module.layers[0](tt_x)
    tt_y = torch.Tensor(ttnn_to_torch(tt_y)).float()
    print("layer0 PCC", pcc(ref_y, tt_y))
    assert pcc(ref_y, tt_y) > 0.98


def test_chemberta_pooler(seq_len=64):
    ref, tt, ids = _make_pair(seq_len, seed=3)
    ref_h = ref.embeddings(ids)
    for layer in ref.encoder:
        ref_h = layer(ref_h)
    ref_p = ref.pooler(ref_h)
    pos = tt_aff._position_ids(ids, CFG["pad_token_id"])
    tokens_tt = _to_tt_ids(tt, ids)
    pos_tt = _to_tt_ids(tt, pos)
    tt_h = tt.module.embed(tokens_tt, pos_tt)
    for layer in tt.module.layers:
        tt_h = layer(tt_h)
    tt_p = tt.module.pooler(tt_h)
    tt_p = torch.Tensor(ttnn_to_torch(tt_p)).float()
    print("pooler PCC", pcc(ref_p, tt_p))
    assert pcc(ref_p, tt_p) > 0.98


def test_chemberta_full_random(seq_len=64):
    ref, tt, ids = _make_pair(seq_len, seed=4)
    ref_pool, ref_hid = ref(ids)
    tt_pool, tt_hid = tt(ids)
    print("full pool PCC", pcc(ref_pool, tt_pool),
          "hid PCC", pcc(ref_hid, tt_hid))
    assert pcc(ref_pool, tt_pool) > 0.98
    assert pcc(ref_hid, tt_hid) > 0.98


def test_chemberta_real_weights(seq_len=64):
    """Real ChemBERTa-zinc-base-v1 weights, random token ids."""
    sd = torch.load(os.environ.get("CHEMBERTA_WEIGHTS", "/tmp/chemberta/pytorch_model.bin"),
                    map_location="cpu", weights_only=False)
    remapped = tt_aff.remap_chemberta_state_dict(sd)
    ref = ChemBERTaReference(CFG).eval()
    # build a ref state_dict in the reference layout from the SAME remapped dict
    ref_sd = {}
    for k, v in remapped.items():
        ref_sd[k.replace("layer.", "encoder.", 1) if k.startswith("layer.") else k] = v
    ref.load_state_dict(ref_sd, strict=False)
    tt = tt_aff.ChemBERTa(CFG)
    tt.load_state_dict(remapped, strict=False)
    ids = torch.randint(5, CFG["vocab_size"], (2, seq_len))
    ref_pool, _ = ref(ids)
    tt_pool, _ = tt(ids)
    print("real pool PCC", pcc(ref_pool, tt_pool))
    assert pcc(ref_pool, tt_pool) > 0.98


# helpers ------------------------------------------------------------------

def _to_tt_ids(wrapper, ids):
    import ttnn
    return ttnn.from_torch(ids.to(torch.int32), device=wrapper.tt_device,
                           layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)


def ttnn_to_torch(t):
    import ttnn
    return ttnn.to_torch(t)


if __name__ == "__main__":
    import ttnn  # noqa
    # standalone runner (one device context) for quick on-box verification
    for name, fn in [
        ("fusion_head_real_weights", test_fusion_head_real_weights),
        ("chemberta_embeddings", test_chemberta_embeddings),
        ("chemberta_layer", test_chemberta_layer),
        ("chemberta_pooler", test_chemberta_pooler),
        ("chemberta_full_random", test_chemberta_full_random),
        ("chemberta_real_weights", test_chemberta_real_weights),
    ]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            print(f"FAIL {name}: {e!r}")
    from tt_bio.tenstorrent import cleanup
    cleanup()

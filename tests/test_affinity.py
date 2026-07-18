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




# ---------------------------------------------------------------------------
# ProtBERT protein encoder (pass 2) — real-weight component parity
# ---------------------------------------------------------------------------

from affinity_reference import (  # noqa: E402
    ProtBERTReference,
    PROTBERT_CFG,
    tokenize_protein as _tokenize_protein_ref,
)

PROTBERT_WEIGHTS = os.environ.get(
    "PROTBERT_WEIGHTS",
    "/home/ttuser/.cache/huggingface/hub/models--Rostlab--prot_bert/"
    "snapshots/7a894481acdc12202f0a415dd567f6cfdb698908/pytorch_model.bin",
)


def _prot_ref_to_tt_keys(sd: dict) -> dict:
    """ProtBERTReference state_dict (encoder.layer.* layout) -> tt layout (layer.*)."""
    import collections
    out = collections.OrderedDict()
    for k, v in sd.items():
        if k.startswith("encoder."):
            out[k.replace("encoder.", "layer.", 1)] = v
        else:
            out[k] = v
    return out


def _load_protbert_real():
    """Load real ProtBERT weights into both the PyTorch reference and the ttnn port."""
    sd = torch.load(PROTBERT_WEIGHTS, map_location="cpu", weights_only=False)
    remapped = tt_aff.remap_protbert_state_dict(sd)
    ref = ProtBERTReference(PROTBERT_CFG).eval()
    ref_sd = {}
    for k, v in remapped.items():
        ref_sd[k.replace("layer.", "encoder.", 1) if k.startswith("layer.") else k] = v
    ref.load_state_dict(ref_sd, strict=False)
    tt = tt_aff.ProtBERT(PROTBERT_CFG)
    tt.load_state_dict(remapped, strict=False)
    return ref, tt


def _prot_ids(seq_len=64, seq=None):
    if seq is None:
        seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ids = _tokenize_protein_ref(seq)
    return ids


def test_protbert_embeddings():
    ref, tt = _load_protbert_real()
    ids = _prot_ids()
    ref_x = ref.embeddings(ids)
    L = ids.shape[1]
    pos = torch.arange(L, dtype=torch.long).unsqueeze(0).expand_as(ids)
    tokens_tt = _to_tt_ids(tt, ids)
    pos_tt = _to_tt_ids(tt, pos)
    tt_x = tt.module.embed(tokens_tt, pos_tt)
    tt_x = torch.Tensor(ttnn_to_torch(tt_x)).float()
    print("prot embed PCC", pcc(ref_x, tt_x))
    assert pcc(ref_x, tt_x) > 0.98


def test_protbert_layer():
    ref, tt = _load_protbert_real()
    ids = _prot_ids()
    ref_x = ref.embeddings(ids)
    ref_y = ref.encoder[0](ref_x)
    L = ids.shape[1]
    pos = torch.arange(L, dtype=torch.long).unsqueeze(0).expand_as(ids)
    tokens_tt = _to_tt_ids(tt, ids)
    pos_tt = _to_tt_ids(tt, pos)
    tt_x = tt.module.embed(tokens_tt, pos_tt)
    tt_y = tt.module.layers[0](tt_x)
    tt_y = torch.Tensor(ttnn_to_torch(tt_y)).float()
    print("prot layer0 PCC", pcc(ref_y, tt_y))
    assert pcc(ref_y, tt_y) > 0.98


def test_protbert_pooler():
    ref, tt = _load_protbert_real()
    ids = _prot_ids()
    ref_h = ref.embeddings(ids)
    for layer in ref.encoder:
        ref_h = layer(ref_h)
    ref_p = ref.pooler(ref_h)
    L = ids.shape[1]
    pos = torch.arange(L, dtype=torch.long).unsqueeze(0).expand_as(ids)
    tokens_tt = _to_tt_ids(tt, ids)
    pos_tt = _to_tt_ids(tt, pos)
    tt_h = tt.module.embed(tokens_tt, pos_tt)
    for layer in tt.module.layers:
        tt_h = layer(tt_h)
    tt_p = tt.module.pooler(tt_h)
    tt_p = torch.Tensor(ttnn_to_torch(tt_p)).float()
    print("prot pooler PCC", pcc(ref_p, tt_p))
    assert pcc(ref_p, tt_p) > 0.98


def test_protbert_full_real():
    ref, tt = _load_protbert_real()
    ids = _prot_ids()
    ref_pool, ref_hid = ref(ids)
    tt_pool, tt_hid = tt(ids)
    print("prot full pool PCC", pcc(ref_pool, tt_pool),
          "hid PCC", pcc(ref_hid, tt_hid))
    assert pcc(ref_pool, tt_pool) > 0.98
    assert pcc(ref_hid, tt_hid) > 0.97  # 30-layer bf16 accumulation on last_hidden


# ---------------------------------------------------------------------------
# SMILES tokenizer (pure-python BPE vs HF RobertaTokenizer)
# ---------------------------------------------------------------------------

def test_smiles_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception:
        pytest.skip("transformers unavailable; BPE parity needs HF reference")
    ref_tok = AutoTokenizer.from_pretrained(str(tt_aff._VENDOR / "chemberta"))
    smiles = [
        "CC(=O)C", "c1ccccc1", "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "CC(=O)Nc1ccc(O)cc1", "O=C(O)c1ccccc1O", "[Na+].[Cl-]", "CC(C)CC(NC(=O)C)C(=O)O",
    ]
    for sm in smiles:
        ref_ids = ref_tok(sm, max_length=278, truncation=True)["input_ids"]
        tt_ids = tt_aff.tokenize_smiles(sm)[0].tolist()
        assert ref_ids == tt_ids, f"SMILES BPE mismatch for {sm}: {ref_ids} vs {tt_ids}"


# ---------------------------------------------------------------------------
# End-to-end pipeline (real weights) vs HF ProtBERT + HF ChemBERTa + ref head
# ---------------------------------------------------------------------------

def test_e2e_affinity():
    """Real-weight end-to-end pKd parity: ttnn pipeline vs HF ProtBERT + HF
    ChemBERTa + the from-scratch fusion head (verified == ONNX in pass 1)."""
    try:
        from transformers import BertModel, RobertaModel, AutoTokenizer
    except Exception:
        pytest.skip("transformers unavailable; e2e parity needs HF reference")
    from affinity_reference import FusionHead as RefFusionHead, AFFINITY_MEAN, AFFINITY_SCALE

    hf_prot = BertModel.from_pretrained("Rostlab/prot_bert").eval()
    hf_mol = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").eval()
    ref_head = RefFusionHead().eval()
    p_tok = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    m_tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    tt = tt_aff.Affinity.from_pretrained()

    pairs = [
        ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
         "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),
        ("MTKIVELQGSGMTVQARLKEACNRSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
         "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),
        ("MSKGYNIVATPRGYVLAGGKIVDQALAQALRLGYNIVATPRGYVLAGGMKTVRQERLKSIVRILERS",
         "CC(=O)Nc1ccc(O)cc1"),
    ]
    for seq, sm in pairs:
        p_ids = p_tok(tt_aff.preprocess_protein(seq), max_length=3200, truncation=True,
                      return_tensors="pt")["input_ids"]
        m_ids = m_tok(sm, max_length=278, truncation=True, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            r_pp = hf_prot(p_ids).pooler_output
            r_mp = hf_mol(m_ids).pooler_output
            r_norm = ref_head(r_pp, r_mp)
        r_pkd = float((r_norm * AFFINITY_SCALE + AFFINITY_MEAN).reshape(-1)[0].item())

        t_pkd = tt.predict(seq, sm)["neg_log10_affinity_M"]
        # bf16 inference budget: pooler PCC >0.9997, pKd within 0.3 of fp32 ref.
        print(f"e2e pKd ref={r_pkd:.4f} tt={t_pkd:.4f} |d|={abs(r_pkd - t_pkd):.4f}")
        assert abs(r_pkd - t_pkd) < 0.3, f"pKd gap {abs(r_pkd - t_pkd):.4f} exceeds bf16 budget"
        # pooler parity (the SACRED part)
        t_pp, _ = tt.prot(tt_aff.tokenize_protein(seq))
        t_mp, _ = tt.mol(tt_aff.tokenize_smiles(sm))
        assert pcc(r_pp, t_pp) > 0.999
        assert pcc(r_mp, t_mp) > 0.999


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
        ("protbert_embeddings", test_protbert_embeddings),
        ("protbert_layer", test_protbert_layer),
        ("protbert_pooler", test_protbert_pooler),
        ("protbert_full_real", test_protbert_full_real),
        ("smiles_tokenizer", test_smiles_tokenizer),
        ("e2e_affinity", test_e2e_affinity),
    ]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            print(f"FAIL {name}: {e!r}")
    from tt_bio.tenstorrent import cleanup
    cleanup()

"""On-device parity for the OpenFold3 InputEmbedder atom-encoder leg:
AtomTransformer (3-block windowed DiT) + atom->token aggregation.

Golden: ~/of3_ref_out.pkl["intermediates"]["input_embedder_atom_transformer_real"],
captured by scripts/of3_atom_transformer_golden.py. The golden carries the
host-precomputed block-gather artifacts (key_block_idxs, invalid_mask, mask_trunked)
and the atom->token mean matrix, so the device AtomTransformer is gated against the
exact reference block structure (the mask-derived gather is captured, not re-derived --
same discipline as the RefAtomFeatureEmbedder golden).

This closes the InputEmbedder atom-encoder leg: cl + plm (PCC-gated in
test_openfold3_ref_atom_feat.py) -> AtomTransformer -> ql -> relu(linear_q) + mean
aggregation -> ai; ai then concatenates with [restype, profile, deletion_mean] to form
s_input (the glue leg, already gated in test_openfold3_input_embedder.py, consumes
s_input). Here ql and ai are gated independently.
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="of3 ckpt or golden pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def test_of3_atom_transformer_on_device():
    """Device OF3AtomTransformer (-> ql) + aggregation (-> ai) vs the reference on real
    ubiquitin. ql is the 3-block windowed-attention output; ai is relu(linear_q(ql))
    mean-aggregated to tokens. Both gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN
    from tt_bio.openfold3_atom_transformer import OF3AtomTransformer
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    enc = _sub(sd, "input_embedder.atom_attn_enc")
    at_sd = _sub(enc, "atom_transformer")
    lq_w = _sub(enc, "linear_q")["0.weight"]  # (384, 128); aggregation linear (ReLU follows)

    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["input_embedder_atom_transformer_real"]
    n_atom, n_token = g["n_atom"], g["n_token"]
    nb, NP = g["nb"], g["NP"]
    s_full, z, ql_ref, ai_ref = g["s_full"], g["z"], g["ql_ref"], g["ai_ref"]
    atom_mask, M = g["atom_mask"], g["atom_to_token_mean"]
    key_block_idxs, invalid_mask, mask_trunked = g["key_block_idxs"], g["invalid_mask"], g["mask_trunked"]

    dev = get_device()
    at = OF3AtomTransformer(at_sd, _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    s_pad = torch.zeros(1, NP, 128); s_pad[0, :n_atom] = s_full
    a_pad = s_pad.clone()  # a_init = cl (rl=None in the InputEmbedder encoder path)
    atom_mask_col = torch.zeros(1, NP, 1); atom_mask_col[0, :n_atom, 0] = atom_mask
    valid = (~invalid_mask).float().reshape(1, nb, 128, 1)
    mask_bias = (1e9 * (mask_trunked - 1)).reshape(1, nb, 1, 32, 128)
    kidx = key_block_idxs.reshape(1, nb * 128).to(torch.int32)

    kidx_tt = ttnn.from_torch(kidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    ql_d = at(ft(a_pad), ft(s_pad), ft(z.unsqueeze(0)), ft(atom_mask_col),
              kidx_tt, ft(valid), ft(mask_bias), n_atom, NP, nb)

    ql = torch.Tensor(ttnn.to_torch(ql_d)).float().reshape(ql_ref.shape)
    ql_pcc = _pcc(ql, ql_ref.float())
    print(f"\nOF3 AtomTransformer: ql_pcc={ql_pcc:.5f}")
    assert ql_pcc > 0.98, f"ql_pcc={ql_pcc:.5f} below 0.98"

    # Aggregation on device: relu(linear_q(ql * atom_mask)) -> matmul(atom_to_token_mean).
    am_col = atom_mask.reshape(1, n_atom, 1)
    ql_masked = ttnn.multiply(ql_d, ft(am_col))
    lq_w_tt = ttnn.from_torch(lq_w.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                              device=dev, dtype=ttnn.bfloat16)
    q_d = ttnn.linear(ql_masked, lq_w_tt, activation="relu",
                      compute_kernel_config=_cfg(dev), core_grid=CORE_GRID_MAIN)
    ai_d = ttnn.matmul(ft(M.unsqueeze(0)), q_d, compute_kernel_config=_cfg(dev))
    ai = torch.Tensor(ttnn.to_torch(ai_d)).float().reshape(ai_ref.shape)
    ai_pcc = _pcc(ai, ai_ref.float())
    print(f"OF3 AtomTransformer aggregation: ai_pcc={ai_pcc:.5f}")
    assert ai_pcc > 0.98, f"ai_pcc={ai_pcc:.5f} below 0.98"

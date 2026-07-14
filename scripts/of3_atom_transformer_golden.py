"""P7: extend ~/of3_ref_out.pkl with the OF3 atom_transformer host-precomputed
block-gather artifacts + atom->token mean matrix, so the device OF3AtomTransformer
is PCC-gated in isolation. The mask-derived block gather (OF3
``convert_single_rep_to_blocks``: centered key windows with underflow/overflow shift)
runs on host here; the device replays it via ``ttnn.embedding`` with the fixed gather
indices -- same isolation discipline as the RefAtomFeatureEmbedder golden
(``dlm``/``vlm``/``inv_sq_dists``).

Adds key ``input_embedder_atom_transformer_real``:
  s_full:      [N_atom, c_atom]      (= cl; the atom conditioning, fixed across blocks)
  z:           [nb, nq, nk, c_atom_pair]  (= plm; the blocked pair, fixed)
  ql_ref:      [N_atom, c_atom]      (reference 3-block atom_transformer output)
  ai_ref:      [N_token, c_token]    (reference atom->token aggregation)
  atom_to_token_mean: [N_token, N_atom]  (mean aggregation matrix)
  key_block_idxs: [nb, nk] int64     (host gather indices for key blocks)
  invalid_mask:   [nb, nk] bool      (invalid key positions -> masked to 0 on device)
  mask_trunked:   [nb, nq, nk] float (q-k validity mask -> attention pad bias)
  atom_mask:      [N_atom] float
  n_atom, n_token, nb, nq, nk, pad_right, NP

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_atom_transformer_golden.py
"""
import os, sys, pickle, math
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)
GOLD = os.path.expanduser("~/of3_ref_out.pkl")

N_QUERY, N_KEY = 32, 128


def main():
    from openfold3.core.utils.atom_attention_block_utils import (
        get_block_indices, get_pair_atom_block_mask, get_query_block_padding,
    )

    g = pickle.load(open(GOLD, "rb"))["intermediates"]
    ae = g["input_embedder_atom_enc_real"]
    ai_ref, ql_ref, cl_ref, plm_ref = ae["out"]
    b = ae["in"]
    atom_mask = b["atom_mask"].float()
    atom_to_token_index = b["atom_to_token_index"].long()
    token_mask = b["token_mask"].float()
    n_atom = int(atom_mask.shape[0])
    n_token = int(token_mask.shape[0])
    nb = math.ceil(n_atom / N_QUERY)
    pad_right = get_query_block_padding(n_atom, N_QUERY)
    NP = nb * N_QUERY

    key_block_idxs, invalid_mask = get_block_indices(
        atom_mask=atom_mask, n_query=N_QUERY, n_key=N_KEY, device=torch.device("cpu"))
    mask_trunked = get_pair_atom_block_mask(
        atom_mask=atom_mask, num_blocks=nb, n_query=N_QUERY, n_key=N_KEY,
        pad_len_right_q=pad_right, key_block_idxs=key_block_idxs, invalid_mask=invalid_mask)

    # atom->token mean matrix M[t,a] = atom_mask[a]/count(t) if atom a maps to token t.
    # Matches OF3 aggregate_atom_feat_to_tokens(..., "mean"): ai = M @ atom_feat where
    # atom_feat is already masked (invalid atoms contribute 0 via M).
    counts = torch.zeros(n_token)
    counts.scatter_add_(0, atom_to_token_index, atom_mask)
    M = torch.zeros(n_token, n_atom)
    valid = atom_mask > 0
    M[atom_to_token_index[valid], torch.arange(n_atom)[valid]] = (
        atom_mask[valid] / counts[atom_to_token_index[valid]].clamp_min(1.0))

    rec = {
        "s_full": cl_ref.clone(), "z": plm_ref.clone(),
        "ql_ref": ql_ref.clone(), "ai_ref": ai_ref.clone(),
        "atom_to_token_mean": M, "key_block_idxs": key_block_idxs.long(),
        "invalid_mask": invalid_mask, "mask_trunked": mask_trunked.float(),
        "atom_mask": atom_mask, "n_atom": n_atom, "n_token": n_token,
        "nb": nb, "nq": N_QUERY, "nk": N_KEY, "pad_right": pad_right, "NP": NP,
    }
    gold = pickle.load(open(GOLD, "rb"))
    # Strip any ml_collections ConfigDict -> plain dict so the pkl loads in the device
    # env (no ml_collections there). ConfigDict is not a dict/Mapping subclass, so
    # duck-type on .items(). Idempotent on an already-clean pkl.
    import collections.abc
    def _strip(o):
        if isinstance(o, torch.Tensor):
            return o
        if (isinstance(o, (dict, collections.abc.Mapping))
                or (hasattr(o, "items") and callable(getattr(o, "items"))
                    and hasattr(o, "__getitem__"))):
            return {k: _strip(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_strip(v) for v in o)
        return o
    gold = _strip(gold)
    gold["intermediates"]["input_embedder_atom_transformer_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("added input_embedder_atom_transformer_real: n_atom", n_atom,
          "n_token", n_token, "nb", nb, "NP", NP,
          "ql std", float(ql_ref.std()), "ai std", float(ai_ref.std()))


if __name__ == "__main__":
    main()

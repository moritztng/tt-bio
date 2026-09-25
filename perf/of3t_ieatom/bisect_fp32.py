#!/usr/bin/env python3
"""of3t-ieatom: stage-by-stage rel of the training InputAtomEncoder against the host leg."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> int:
    batch = Path(sys.argv[1])
    import ttnn
    from tt_bio.openfold3_fold import build_dm_device_aux
    from tt_bio.openfold3_host_prep import (N_KEY, N_QUERY, _lin, convert_single_rep_to_blocks,
                                            derive_block_aux, ref_atom_device_inputs,
                                            ref_atom_embed)
    from tt_bio.openfold3_weights import _sub
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward

    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=20260922)
    m, dev = fwd.model, fwd.device
    E = m.input_atom_enc
    f = OpenFold3Dataset(batch).batch([0])["features"]
    aux = derive_block_aux(f)
    n_atom, NP, nb = aux["n_atom"], aux["NP"], aux["nb"]
    enc = _sub(m.sd, "input_embedder.atom_attn_enc")
    cl_h, plm_h = ref_atom_embed(_sub(enc, "ref_atom_feature_embedder"), f)
    cl_l, cl_m, pmask = convert_single_rep_to_blocks(ql=cl_h, n_query=N_QUERY, n_key=N_KEY,
                                                     atom_mask=aux["atom_mask"])
    pu_h = plm_h + (_lin(cl_l.relu().unsqueeze(-2), enc, "linear_l")
                    + _lin(cl_m.relu().unsqueeze(-3), enc, "linear_m")) * pmask.unsqueeze(-1)
    h = pu_h
    for k in ("pair_mlp.1", "pair_mlp.3", "pair_mlp.5"):
        h = _lin(h.relu(), enc, k)
    pu_h = (pu_h + h) * pmask.unsqueeze(-1)

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                   dtype=ttnn.bfloat16)
    tok = f["token_mask"].float()
    cl0, plm0 = ref_atom_embed(
        _sub(m.sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), f)
    a = build_dm_device_aux(
        dev, ft, cl0=cl0, plm0=plm0, atom_mask=aux["atom_mask"],
        atom_to_token_index=aux["atom_to_token_index"],
        npe_q_indices=aux["npe_q_indices"], npe_k_indices=aux["npe_k_indices"],
        zij_mask=aux["zij_mask"], key_block_idxs=aux["key_block_idxs"],
        invalid_mask=aux["invalid_mask"], mask_trunked=aux["mask_trunked"],
        atom_to_token_mean=aux["atom_to_token_mean"], token_mask=tok, n_atom=n_atom,
        n_token=aux["n_token"], nb=nb, NP=NP, n_tok_pad=aux["n_token"])
    dt = E._act_dtype
    c = lambda t: ttnn.typecast(t, dt) if t.dtype != dt else t  # noqa: E731
    th = lambda t: ttnn.to_torch(t).float()  # noqa: E731
    rel = lambda x, y: float((x - y).norm() / y.norm())  # noqa: E731
    ref_in = ref_atom_device_inputs(dev, f, aux["atom_mask"], NP, dtype=dt)
    print("act dtype", dt, "input dtypes", [t.dtype for t in ref_in], flush=True)
    from tt_bio.tenstorrent import device_dtype_override
    ov = device_dtype_override(dt); ov.__enter__()
    cl, plm = E.ref_embed(*ref_in)
    print("weights", E.ref_embed.w_ref_pos.dtype, "cl", cl.dtype, cl.shape, "plm", plm.shape)
    print("ref_embed cl rel", rel(th(cl)[0, :n_atom], cl_h), "plm rel", rel(th(plm)[0], plm_h))
    pu = E.pair_update(cl, plm, a["kidx_tt"], c(a["valid_d"]), c(a["pm_d"]), NP, nb)
    print("pair_update rel", rel(th(pu)[0], pu_h), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def masks(batch):
    """Host only: the diffusion module's blocking masks against the host leg's pair mask."""
    from tt_bio.openfold3_host_prep import (N_KEY, N_QUERY, convert_single_rep_to_blocks,
                                            derive_block_aux)
    from tt_bio.train.openfold3 import OpenFold3Dataset
    f = OpenFold3Dataset(batch).batch([0])["features"]
    aux = derive_block_aux(f)
    x = torch.arange(aux["n_atom"]).float().unsqueeze(-1) + 1
    _l, xm, pm = convert_single_rep_to_blocks(ql=x, n_query=N_QUERY, n_key=N_KEY,
                                              atom_mask=aux["atom_mask"])
    mt = aux["mask_trunked"].reshape(pm.shape).float()
    print("pair mask host vs mask_trunked: shape", tuple(pm.shape), "differ", int((pm != mt).sum()),
          "of", pm.numel(), "host ones", int(pm.sum()), "trunked ones", int(mt.sum()))
    valid = (~aux["invalid_mask"]).float().reshape(xm.shape[:-1])
    print("host cl_m nonzero where invalid:", int(((xm[..., 0] != 0) & (valid == 0)).sum()),
          "zero where valid:", int(((xm[..., 0] == 0) & (valid == 1)).sum()))
    kidx = aux["key_block_idxs"].reshape(xm.shape[:-1]).long()
    g = torch.zeros(x.shape[0] + 1000, 1); g[: x.shape[0]] = x
    gm = g[kidx.clamp(max=g.shape[0] - 1)][..., 0] * valid
    print("gather(key_block_idxs)*valid == host cl_m:", bool(torch.equal(gm, xm[..., 0])))


def stages(batch):
    """Device InputAtomEncoder stage by stage against the host leg, each stage also fed exact
    host inputs so its own error is separated from what it inherits."""
    import ttnn
    from tt_bio.openfold3_atom_transformer import OF3AtomTransformer
    from tt_bio.openfold3_host_prep import (N_KEY, N_QUERY, _lin, convert_single_rep_to_blocks,
                                            derive_block_aux, ref_atom_device_inputs,
                                            ref_atom_embed)
    from tt_bio.openfold3_weights import _sub
    from tt_bio.tenstorrent import device_dtype_override
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward

    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=20260922)
    m, dev = fwd.model, fwd.device
    E = m.input_atom_enc
    dt = E._act_dtype
    f = OpenFold3Dataset(batch).batch([0])["features"]
    aux = derive_block_aux(f)
    n_atom, NP, nb = aux["n_atom"], aux["NP"], aux["nb"]
    am = aux["atom_mask"]
    enc = _sub(m.sd, "input_embedder.atom_attn_enc")
    cl_h, plm_h = ref_atom_embed(_sub(enc, "ref_atom_feature_embedder"), f)
    cl_l, cl_m, pmask = convert_single_rep_to_blocks(ql=cl_h, n_query=N_QUERY, n_key=N_KEY,
                                                     atom_mask=am)
    pu_h = plm_h + (_lin(cl_l.relu().unsqueeze(-2), enc, "linear_l")
                    + _lin(cl_m.relu().unsqueeze(-3), enc, "linear_m")) * pmask.unsqueeze(-1)
    h = pu_h
    for k in ("pair_mlp.1", "pair_mlp.3", "pair_mlp.5"):
        h = _lin(h.relu(), enc, k)
    pu_h = (pu_h + h) * pmask.unsqueeze(-1)
    up = lambda x, d=dt: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)  # noqa: E731
    th = lambda t: ttnn.to_torch(t).float()  # noqa: E731
    rel = lambda x, y: float((x - y).norm() / y.norm())  # noqa: E731
    a_pad = torch.zeros(1, NP, 128); a_pad[0, :n_atom] = cl_h
    amc = torch.zeros(1, NP, 1); amc[0, :n_atom, 0] = am
    valid = (~aux["invalid_mask"]).float().reshape(1, nb, N_KEY, 1)
    mb = (1e9 * (aux["mask_trunked"] - 1)).reshape(1, nb, 1, N_QUERY, N_KEY)
    kidx = ttnn.from_torch(aux["key_block_idxs"].reshape(1, nb * N_KEY).to(torch.int32),
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    pmd = aux["mask_trunked"].reshape(1, nb, N_QUERY, N_KEY, 1)
    # host leg's atom transformer: bf16 module, bf16 inputs, exact host cl / plm
    at16 = OF3AtomTransformer(_sub(enc, "atom_transformer"), m.ckc)
    b = ttnn.bfloat16
    ql_h = th(at16(up(a_pad, b), up(a_pad, b), up(pu_h.unsqueeze(0), b), up(amc, b), kidx,
                   up(valid, b), up(mb, b), n_atom, NP, nb))[0, :n_atom]
    lq = enc["linear_q.0.weight"].float()
    ai_of = lambda ql: aux["atom_to_token_mean"] @ torch.nn.functional.linear(ql * am[:, None], lq).relu()  # noqa: E731
    ai_h = ai_of(ql_h)
    with device_dtype_override(dt):
        ref_in = ref_atom_device_inputs(dev, f, am, NP, dtype=dt)
        cl, plm = E.ref_embed(*ref_in)
        print("ref_embed   cl", rel(th(cl)[0, :n_atom], cl_h), "plm", rel(th(plm)[0], plm_h))
        pu = E.pair_update(cl, plm, kidx, up(valid), up(pmd), NP, nb)
        print("pair_update chained", rel(th(pu)[0], pu_h))
        pu_x = E.pair_update(up(a_pad), up(plm_h.unsqueeze(0)), kidx, up(valid), up(pmd), NP, nb)
        print("pair_update on exact input", rel(th(pu_x)[0], pu_h))
        ql = E.at(cl, cl, pu, up(amc), kidx, up(valid), up(mb), n_atom, NP, nb)
        print("at chained vs host ql", rel(th(ql)[0], ql_h))
        ql_x = E.at(up(a_pad), up(a_pad), up(pu_h.unsqueeze(0)), up(amc), kidx, up(valid),
                    up(mb), n_atom, NP, nb)
        print("fp32 at on exact input vs host bf16 at", rel(th(ql_x)[0], ql_h),
              "| ai from it", rel(ai_of(th(ql_x)[0]), ai_h))
        amn = torch.zeros(1, n_atom, 1); amn[0, :, 0] = am
        ai = E.head(up(ql_h.unsqueeze(0)), up(amn), up(aux["atom_to_token_mean"].unsqueeze(0)))
        print("head on exact ql", rel(th(ai)[0], ai_h))
        ai_c = E.head(ql, up(amn), up(aux["atom_to_token_mean"].unsqueeze(0)))
        print("head chained (full device ai) vs host ai", rel(th(ai_c)[0], ai_h))

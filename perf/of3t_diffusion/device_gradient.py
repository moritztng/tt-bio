#!/usr/bin/env python3
"""Instrument A at model scope: our taped diffusion gradient against the reference's.

Runs OUR `OF3DiffusionModule` on the card at the boundary captured from BUNDLE-MIN's own r = 0
step, seeded with THEIR cotangent, and compares every parameter gradient against the float64
reference taken at the same boundary. D19 is upstream of the boundary, so it cancels: the floor
here is zero rather than the 6.735e-03 forward gap.

Scope: our module is their `diffusion_module` minus `diffusion_conditioning`, which on our side
is a separate class that feeds it -- 712 of 738 tensors, 66.39 % of the diffusion squared norm.
The conditioned `(si, zij)` come from THEIR conditioning via sub_boundary.pt, so nothing our own
conditioning does can flatter this number.

THE BIJECTION IS TAKEN FROM TENSOR IDENTITY, NOT THE LOADER (amendment item 1). `_w_tt`
transposes every weight on host and caches the result, so 598 of 870 device tensors never pass
`Module.torch_to_tt` and a leaf count taken there is 69 % short. Every load site resolves a
torch tensor out of the checkpoint sub-dict first, so `id(that tensor) -> checkpoint name` maps
all of them, and the device gradient is transposed back by comparing its shape to the
checkpoint tensor's before any comparison is made.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_tape"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
CAP = "/home/ttuser/of3t_diffusion_cap"
OUT = "perf/of3t_diffusion"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--structs", default="0",
                   help="'0' for one structure, 'all' for the 48 the bundle gradient sums")
    p.add_argument("--tag", default="")
    p.add_argument("--mask-ones", action="store_true", dest="mask_ones",
                   help="run the DiT with every token real, against their DiT re-run "
                        "the same way, to separate an 85 %%-padding defect from a "
                        "384-token one")
    # D23/R126: this instrument scores against S["xl_out"] in the captured boundary, so which
    # capture it reads is part of the measurement. Defaults are the published 0.5.0 paths.
    p.add_argument("--cap", default=CAP,
                   help="captured diffusion boundary to score against")
    p.add_argument("--out-dir", default=OUT)
    p.add_argument("--dump-per-tensor", action="store_true", dest="dump_per_tensor",
                   help="also inline the per-tensor array in the main report. The array is "
                        "written to a sidecar unconditionally either way; this only controls "
                        "the duplicate copy, kept so the reports that already carry it stay "
                        "comparable.")
    p.add_argument("--dump-grads", default="", dest="dump_grads",
                   help="write the compared device gradient tensors themselves to this .pt, "
                        "keyed by full checkpoint name. Without it this instrument publishes "
                        "only rel_l2/norm_ratio/cos against the one reference it was run "
                        "against, so the gradient cannot afterwards be compared to anything "
                        "else -- including upstream's own bf16 training gradient.")
    p.add_argument("--permute-cot", action="store_true", dest="permute_cot",
                   help="negative control: seed structure k's backward with structure k+1's "
                        "cotangent, so every sample is paired with the wrong one while the "
                        "forward, the weights and the arithmetic are untouched. A comparison "
                        "that cannot tell this apart from the real run is not measuring "
                        "agreement with anything.")
    p.add_argument("--softmax-f64", action="store_true", dest="softmax_f64",
                   help="AMENDMENT 3's BOUND, not a lever: compute every softmax the tape sees, "
                        "forward and backward, on the host in float64. of3t-adaln's rule "
                        "verbatim, installed into taped_ttnn._VERBS so it is scope-agnostic and "
                        "reaches all 24 DiT blocks and both atom transformers. What is left "
                        "after it is what the softmax cannot explain.")
    p.add_argument("--ckc-census", action="store_true", dest="ckc_census",
                   help="AMENDMENT 1's ckc_on arm. Wraps the SHIPPED softmax tape rule with a "
                        "census of the `compute_kernel_config` each call actually passes, and "
                        "changes no arithmetic. Run it with TT_BIO_SOFTMAX_CKC=1 to answer what "
                        "that flag routes into this scope: the flag hands the CALLER's config to "
                        "`ttnn.softmax_in_place` inside `_fp32_softmax_attention`, which is the "
                        "trunk's triangle-attention path, so whether it reaches the diffusion "
                        "module at all is a measurement and not a reading of the source. "
                        "`FP32_SOFTMAX_STATS` is published beside it.")
    p.add_argument("--softmax-lever", default="", dest="softmax_lever",
                   choices=["", "precise", "accurate"],
                   help="a SHIPPABLE softmax lever, unlike --softmax-f64 which is a host "
                        "bound. `precise` adds precise_config() to the forward softmax call "
                        "and leaves the shipped backward alone; `accurate` runs "
                        "tenstorrent._accurate_softmax in the forward and adds "
                        "precise_config() to the backward reduction. Both are of3t-adaln's "
                        "block-8 rules verbatim, installed on the tape verb so they reach "
                        "all 24 DiT blocks and both atom transformers. Mutually exclusive "
                        "with --softmax-f64; the intercept count is published and a zero is "
                        "a hard failure.")
    p.add_argument("--host-f64", default="", dest="host_f64",
                   help="comma-separated tape verbs to compute, forward and backward, on the "
                        "host in float64 -- the same BOUND the softmax got, generalised. "
                        "`layer_norm,multiply,multiply_,add,add_` bounds everything the "
                        "conditioned transition does except its matmuls; adding `linear` "
                        "bounds those too. Scope-agnostic like the softmax bound, so it "
                        "bounds MORE than the transition and is therefore an upper bound on "
                        "what the transition can contribute. Each verb's intercept count is "
                        "published and a zero is a hard failure.")
    p.add_argument("--softmax-site-f64", action="store_true", dest="softmax_site_f64",
                   help="the SUPPORTED host float64 softmax path, selected the way a model "
                        "selects it: `TT_BIO_HOST_F64_SOFTMAX_AB=all`, resolved per construction "
                        "site by `tenstorrent.host_f64_softmax_site`. Unlike --softmax-f64 this "
                        "patches nothing -- the arm runs the shipped call sites with the site "
                        "flag on, so what it measures is the code path rather than a rule "
                        "installed over the tape verb. `HOST_F64_SOFTMAX_STATS` is published and "
                        "a zero served count is a hard failure.")
    p.add_argument("--verb-census", action="store_true", dest="verb_census",
                   help="count every tape verb without changing any arithmetic, so the cost "
                        "of a bound is known before it is paid")
    p.add_argument("--bisect", action="store_true",
                   help="compare every stage against their captured intermediates, "
                        "which localises a forward gap instead of reporting it")
    a = p.parse_args()
    t0 = time.perf_counter()

    # Before the first tt_bio import: `host_f64_softmax_site` is resolved at module CONSTRUCTION,
    # so an environment set after the modules are built decides nothing. The arm names the env
    # var the models read rather than reaching into the selector, because a flag that only this
    # harness can set is not the code path.
    if a.softmax_site_f64:
        os.environ["TT_BIO_HOST_F64_SOFTMAX_AB"] = "all"

    import torch
    import ttnn
    from of3_coverage import _device_weights

    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion_module import OF3DiffusionModule
    from tt_bio.openfold3_weights import _sub
    from tt_bio.openfold3_fold import build_dm_device_aux
    from tt_bio import openfold3_host_prep as HP
    from tt_bio._vendor.openfold3.core.utils.atom_attention_block_utils import (
        get_block_indices, get_pair_atom_block_mask, get_query_block_padding)
    import torch.nn.functional as F

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dmsd = _sub(sd, "diffusion_module")
    name_by_id = {id(v): k for k, v in dmsd.items() if torch.is_tensor(v)}
    shape_by_name = {k: tuple(v.shape) for k, v in dmsd.items() if torch.is_tensor(v)}
    print(f"[{time.perf_counter()-t0:.0f}s] checkpoint: {len(name_by_id)} tensors under "
          f"diffusion_module.", flush=True)

    # ---- the bijection: fingerprint, so no load path can be missed ---------------------------
    # The previous version patched three named load sites and reached 283 of 870 device
    # weights, because three more classes load by paths those three do not cover. Patching
    # `ttnn.from_torch` for the duration of construction cannot miss a path by construction,
    # and the tensor it is handed is matched back to the checkpoint by a transpose-invariant
    # fingerprint rather than by identity -- `_w_tt` hands it `w.t().contiguous()`, a fresh
    # object whose id maps to nothing.
    def fingerprint(x):
        x = x.double()
        return (tuple(sorted(x.shape)), round(float(x.sum()), 9),
                round(float(x.abs().max()), 9), x.numel())

    fp_name, fp_clash = {}, set()
    for k, v in dmsd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        f = fingerprint(v)
        if f in fp_name:
            fp_clash.add(f)
        fp_name[f] = k

    # D83. THE ORIENTATION IS RECORDED HERE, NOT INFERRED FROM SHAPE AT COMPARISON TIME.
    # `_w_tt` puts `w.t()` on the card, so the taped dW rule returns the gradient in the
    # device's orientation and the comparison has to put it back. It used to do that with
    # `if shape != want and shape[::-1] == want: gt = gt.t()`, which CANNOT FIRE ON A SQUARE
    # WEIGHT -- and 87 of the 547 compared tensors are square. Every one of them was scored
    # as dW against dW^T: rel_l2 1.40 (= sqrt(2)), norm ratio 1.00, cosine 0.002, in every
    # arm, unmoved by any precision lever, because a transpose is norm-preserving and
    # decorrelating. That was 56.03 % of the float64-softmax bound's squared error and none
    # of it was the port's. The fingerprint is deliberately transpose-invariant, so it could
    # never have caught this either. Here the tensor handed to `ttnn.from_torch` is compared
    # to the checkpoint tensor BIT-EXACTLY in both orientations, so the answer is a fact
    # about the load and not an inference from a shape.
    reg, orient = {}, {}
    orig_from_torch = ttnn.from_torch

    def recording_from_torch(tensor, *args, **kw):
        v = orig_from_torch(tensor, *args, **kw)
        try:
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                f = fingerprint(tensor)
                if f in fp_name and f not in fp_clash:
                    nm = fp_name[f]
                    reg[id(v)] = nm
                    ck = dmsd[nm]
                    same = tensor.shape == ck.shape and bool(torch.equal(tensor, ck))
                    flip = (tensor.dim() == 2 and ck.dim() == 2
                            and tensor.shape == ck.t().shape
                            and bool(torch.equal(tensor, ck.t().contiguous())))
                    # A symmetric square weight satisfies both and the orientation does not
                    # matter; anything that satisfies neither (a reshape, a pad, a fuse) is
                    # recorded as unknown and falls back to the shape test, which is correct
                    # for it because it is not a plain transpose.
                    orient[id(v)] = "N" if same else ("T" if flip else "?")
        except Exception:
            pass
        return v

    # ---- the reference boundary ---------------------------------------------------------------
    B = torch.load(f"{a.cap}/diffusion_boundary.pt", map_location="cpu", weights_only=False)
    S = torch.load(f"{a.cap}/sub_boundary.pt", map_location="cpu", weights_only=False)
    kw, cot, ref_grad = B["kwargs"], B["cot"], S["grad_f64"]
    si_all, zij_ref = S["cond_out"][0], S["cond_out"][1]
    batch = kw["batch"]
    sq = lambda x: x.reshape(x.shape[2:]) if x.dim() > 2 and x.shape[0] == 1 and x.shape[1] == 1 \
        else x.squeeze(0)
    n_atom = int(kw["atom_mask"].shape[-1])
    n_token = int(kw["token_mask"].shape[-1])
    N_STRUCT = int(kw["xl_noisy"].shape[1])
    sigma_data = float(dmsd.get("diffusion_module.sigma_data", torch.tensor(16.0)))
    print(f"[{time.perf_counter()-t0:.0f}s] boundary: {n_token} tokens, {n_atom} atoms, "
          f"N={N_STRUCT}, sigma_data={sigma_data}, cot {tuple(cot.shape)}", flush=True)

    atom_mask = sq(batch["atom_mask"]).float()
    a2t = sq(batch["atom_to_token_index"]).long()
    token_mask = sq(batch["token_mask"]).float()
    N_QUERY, N_KEY = 32, 128
    nb = math.ceil(n_atom / N_QUERY)
    NP = nb * N_QUERY
    n_tok_pad = math.ceil(n_token / 32) * 32
    pad_right = get_query_block_padding(n_atom, N_QUERY)
    key_block_idxs, invalid_mask = get_block_indices(
        atom_mask=atom_mask, n_query=N_QUERY, n_key=N_KEY, device=torch.device("cpu"))
    mask_trunked = get_pair_atom_block_mask(
        atom_mask=atom_mask, num_blocks=nb, n_query=N_QUERY, n_key=N_KEY,
        pad_len_right_q=pad_right, key_block_idxs=key_block_idxs, invalid_mask=invalid_mask)
    npe_q = F.pad(a2t, (0, pad_right), value=0).reshape(nb, N_QUERY).long()
    npe_k = torch.gather(a2t.unsqueeze(0).expand(nb, n_atom), 1, key_block_idxs.long())
    zij_mask = ((~invalid_mask).float())[:, None, :].expand(nb, N_QUERY, N_KEY) * mask_trunked
    a2t_mean = torch.zeros(n_token, n_atom)
    a2t_mean[a2t, torch.arange(n_atom)] = atom_mask
    a2t_mean = a2t_mean / a2t_mean.sum(-1, keepdim=True).clamp_min(1.0)

    feats = {"ref_pos": sq(batch["ref_pos"]).float(), "atom_mask": atom_mask,
             "ref_charge": sq(batch["ref_charge"]).float(),
             "ref_mask": sq(batch["ref_mask"]).float(),
             "ref_element": sq(batch["ref_element"]).float(),
             "ref_atom_name_chars": sq(batch["ref_atom_name_chars"]).float(),
             "ref_space_uid": sq(batch["ref_space_uid"]).float()}
    enc_sd = _sub(dmsd, "atom_attn_enc")
    rafe = _sub(enc_sd, "ref_atom_feature_embedder")
    if not rafe:
        print("ref_atom_feature_embedder keys absent; encoder keys:",
              sorted(set(k.split(".")[0] for k in enc_sd))[:20], flush=True)
        return 1
    cl0, plm0 = HP.ref_atom_embed(rafe, feats)
    print(f"[{time.perf_counter()-t0:.0f}s] host prep: nb={nb} NP={NP} n_tok_pad={n_tok_pad}, "
          f"cl0 {tuple(cl0.shape)} plm0 {tuple(plm0.shape)}", flush=True)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.float32
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)

    # AMENDMENT 3's bound. of3t-adaln's `host_f64_rule` verbatim, plus a fire counter: the
    # shim caches its wrapper, so a patch that does not drop it is a silent no-op, and that is
    # exactly how the first sandwich arm reported two identical columns. An arm that reads the
    # same as the shipped one because nothing fired is not a bound, so the count is published
    # and a zero count is a hard failure.
    softmax_calls = [0]
    trace = []
    fwd_trace = []
    ckc_seen = {}
    if a.ckc_census:
        from tt_bio import taped_ttnn as TT
        import tt_bio.tenstorrent as _TT

        _shipped_sm = TT._VERBS["softmax"]

        def ckc_census_rule(shipped, args, kwargs):
            softmax_calls[0] += 1
            k = repr(kwargs.get("compute_kernel_config"))
            ckc_seen[k] = ckc_seen.get(k, 0) + 1
            return _shipped_sm(shipped, args, kwargs)

        TT._VERBS["softmax"] = ckc_census_rule
        TT._SHIM.__dict__.pop("softmax", None)
        print(f"[{time.perf_counter()-t0:.0f}s] ckc census installed; "
              f"_SOFTMAX_CKC={_TT._SOFTMAX_CKC}", flush=True)
    if a.softmax_f64:
        from tt_bio import taped_ttnn as TT

        def host_f64_softmax(shipped, args, kwargs):
            softmax_calls[0] += 1
            x = TT._wrap(args[0])
            dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
            y64 = torch.softmax(ttnn.to_torch(x.value).double(), dim=dim)
            y0 = ttnn.from_torch(y64.float(), layout=x.value.layout, device=dev,
                                 dtype=x.value.dtype)

            def make():
                def bw(g):
                    g64 = ttnn.to_torch(g).double()
                    d = y64 * (g64 - (g64 * y64).sum(dim=dim, keepdim=True))
                    x.add_grad(ttnn.from_torch(d.float(), layout=g.layout, device=dev,
                                               dtype=g.dtype))
                return bw

            return TT._tape(y0, [x], make)

        TT._VERBS["softmax"] = host_f64_softmax
        TT._SHIM.__dict__.pop("softmax", None)
        print(f"[{time.perf_counter()-t0:.0f}s] float64 softmax bound installed on the tape verb",
              flush=True)

    # The two levers that could actually ship. of3t-adaln scored these on ONE block; at scope
    # they have never been scored at all, and one block is not the arm. Same counter, same
    # zero-count hard failure: a lever that fires nothing is the shipped arm under another
    # name, which is exactly how the first sandwich arm reported two identical columns.
    if a.softmax_lever:
        if a.softmax_f64:
            print("FAILED: --softmax-lever and --softmax-f64 are mutually exclusive; one is a "
                  "shippable lever and the other is a host bound", flush=True)
            return 2
        from tt_bio import taped_ttnn as TT
        from tt_bio.autograd import precise_config
        from tt_bio.tenstorrent import _accurate_softmax, _SOFTMAX_PRECISE_CKC
        probe = os.environ.get("OF3T_SOFTGRAD_PROBE") == "1"
        bad = []
        shipped_softmax_rule = TT._VERBS["softmax"]
        # of3t-nanfloor: the config on the CHAIN's forward reduction, made A/B-able instead of
        # assumed. This row was dispatched on the premise that the published
        # `softmax_accurate` arm ran that reduction at the op default; it has passed
        # `_SOFTMAX_PRECISE_CKC` since the arm was written, so the only way to say what that
        # argument is worth at scope is to run the counterfactual. The default is the shipped
        # arm unchanged, and the census records what reached the op rather than what the
        # source says (D110).
        _sum_ckc = (None if os.environ.get("OF3T_NANFLOOR_SUM_CKC") == "none"
                    else _SOFTMAX_PRECISE_CKC)
        sum_ckc_census = {"fwd_precise": 0, "fwd_none": 0, "bw_precise": 0}

        def precise_forward_rule(shipped, args, kwargs):
            """precise_config() on the FORWARD softmax call, backward untouched.

            of3t-adaln's rule wrote this with `kwargs.setdefault(...)`, and every softmax call
            site in this path passes `compute_kernel_config=self._softmax_ckc` EXPLICITLY
            (openfold3_diffusion_transformer.py:209, openfold3_atom_transformer.py:184).
            `softmax_ckc` returns None when the site flag is off, so the key is present with
            value None and `setdefault` cannot write it: the arm fired 1,440 times and returned
            all 547 gradient tensors bit-identical to shipped. Assignment, not setdefault, and
            the config is the shipped `_SOFTMAX_PRECISE_CKC` rather than a second copy of it."""
            softmax_calls[0] += 1
            kwargs = dict(kwargs)
            kwargs["compute_kernel_config"] = _SOFTMAX_PRECISE_CKC
            return shipped_softmax_rule(shipped, args, kwargs)

        def accurate_forward_rule(shipped, args, kwargs):
            """of3t-adaln's `accurate_forward_rule`: _accurate_softmax in the forward, the
            shipped backward with precise_config() on its reduction."""
            softmax_calls[0] += 1
            x = TT._wrap(args[0])
            dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
            # `_accurate_softmax` reduces over the last axis only. A call on any other axis
            # would silently become a MIXED arm, so it throws rather than reporting.
            nd = len(x.value.shape)
            if dim not in (-1, nd - 1):
                raise AssertionError(
                    f"--softmax-lever accurate got dim={dim} on a rank-{nd} tensor; "
                    f"_accurate_softmax only reduces the last axis")
            # `_accurate_softmax` verbatim returns NaN here, and the reason belongs where the
            # arm is defined. `ttnn.max` rounds its result to bf16, so on a row whose keys are
            # ALL masked (-1e9, which the atom encoder block-sparse attention produces for a
            # padded query block) the reduction comes back 1.76e6 ABOVE the true row max. Every
            # exponent is then -1.76e6, `exp` underflows the whole row to zero, the sum is zero
            # and the divide is 0/0. Measured standalone in FULLY_MASKED_ROW_OVERFLOW.json:
            # 114,688 non-finite entries, exactly the masked rows, where the fused kernel on the
            # same input is finite.
            # The repair is one op. Clamping the exponent at -60 is a no-op where the max is
            # right -- exp(-60) is 8.8e-27 against a row sum of 1 -- and turns a fully-masked
            # row into a uniform distribution, which is masked out downstream anyway. -88 does
            # NOT work: exp(-88) is subnormal in fp32 and flushes to zero, so the sum is zero
            # again. This is the chain a shipped version would have to carry.
            xv = x.value
            _m = ttnn.max(xv, dim=-1, keepdim=True)
            _d = ttnn.subtract(xv, _m)
            ttnn.deallocate(_m)
            _d = ttnn.maximum(_d, -60.0)
            ttnn.exp(_d, output_tensor=_d)
            _s = ttnn.sum(_d, dim=-1, keepdim=True, compute_kernel_config=_sum_ckc)
            sum_ckc_census["fwd_precise" if _sum_ckc is not None else "fwd_none"] += 1
            y0 = ttnn.divide(_d, _s)
            ttnn.deallocate(_d)
            ttnn.deallocate(_s)
            if y0.dtype != xv.dtype:
                _q = ttnn.typecast(y0, xv.dtype, memory_config=y0.memory_config())
                ttnn.deallocate(y0)
                y0 = _q
            n = softmax_calls[0]
            if probe:
                xf = torch.isfinite(ttnn.to_torch(x.value)).all().item()
                yf = torch.isfinite(ttnn.to_torch(y0)).all().item()
                if not (xf and yf):
                    bad.append(f"fwd call {n} shape {tuple(x.value.shape)} "
                               f"x_finite={xf} y_finite={yf}")
            box = [y0]

            def make():
                def bw(g):
                    y = box[0]
                    inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True,
                                     compute_kernel_config=_SOFTMAX_PRECISE_CKC)
                    sum_ckc_census["bw_precise"] += 1
                    d = ttnn.multiply(y, ttnn.subtract(g, inner))
                    if probe:
                        gf = torch.isfinite(ttnn.to_torch(g)).all().item()
                        yf = torch.isfinite(ttnn.to_torch(y)).all().item()
                        inf_ = torch.isfinite(ttnn.to_torch(inner)).all().item()
                        df = torch.isfinite(ttnn.to_torch(d)).all().item()
                        if not (gf and yf and inf_ and df):
                            bad.append(f"bw call {n} shape {tuple(y.shape)} g={gf} y={yf} "
                                       f"inner={inf_} out={df}")
                    x.add_grad(d)
                return bw

            out = TT._tape(y0, [x], make)
            if out.node is not None:
                out.box = box
            return out

        TT._VERBS["softmax"] = {"precise": precise_forward_rule,
                                "accurate": accurate_forward_rule}[a.softmax_lever]
        TT._SHIM.__dict__.pop("softmax", None)

        if os.environ.get("OF3T_SOFTGRAD_PROBE") in ("2", "3"):
            # Wrap every verb, softmax included, and record (verb, whether the gradient handed
            # to its closure was finite) in backward execution order. The first False names the
            # verb whose CONSUMER produced the infinity, and the entries before it are the last
            # healthy steps, so the pair localises the creation to one op.
            fwdprobe = os.environ.get("OF3T_SOFTGRAD_PROBE") == "3"

            def wrap(name, rule):
                def impl(shipped, args, kwargs):
                    out = rule(shipped, args, kwargs)
                    if fwdprobe:
                        try:
                            v = getattr(out, "value", out)
                            shp = tuple(v.shape)
                            okf = bool(torch.isfinite(ttnn.to_torch(v)).all().item())
                            fwd_trace.append((name, okf, shp))
                        except Exception:
                            pass
                    nd = getattr(out, "node", None)
                    if nd is not None and getattr(nd, "fn", None) is not None:
                        inner_fn = nd.fn

                        def checked(g, _n=name, _f=inner_fn):
                            ok = bool(torch.isfinite(ttnn.to_torch(g)).all().item())
                            trace.append((_n, ok, tuple(g.shape)))
                            return _f(g)

                        nd.fn = checked
                    return out
                return impl

            for _nm, _rule in list(TT._VERBS.items()):
                TT._VERBS[_nm] = wrap(_nm, _rule)
                TT._SHIM.__dict__.pop(_nm, None)
            print("PROBE2 armed on", len(TT._VERBS), "verbs", flush=True)
        print(f"[{time.perf_counter()-t0:.0f}s] softmax lever '{a.softmax_lever}' installed on "
              f"the tape verb", flush=True)

    # The same bound on the other verbs, in this row's own namespace so the shared instrument
    # keeps one hook rather than six rules.
    hf64_verbs = [v for v in a.host_f64.split(",") if v]
    if hf64_verbs or a.verb_census:
        sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_residual"))
        import host_f64 as HF
        if a.verb_census:
            HF.census_install()
            print(f"[{time.perf_counter()-t0:.0f}s] verb census installed", flush=True)
        if hf64_verbs:
            HF.install(hf64_verbs)
            print(f"[{time.perf_counter()-t0:.0f}s] float64 bound installed on "
                  f"{','.join(hf64_verbs)}", flush=True)

    aux = build_dm_device_aux(
        dev, ft, cl0=cl0, plm0=plm0, atom_mask=atom_mask, atom_to_token_index=a2t,
        npe_q_indices=npe_q, npe_k_indices=npe_k, zij_mask=zij_mask,
        key_block_idxs=key_block_idxs, invalid_mask=invalid_mask, mask_trunked=mask_trunked,
        atom_to_token_mean=a2t_mean, token_mask=token_mask, n_atom=n_atom, n_token=n_token,
        nb=nb, NP=NP, n_tok_pad=n_tok_pad)
    if a.mask_ones:
        ones = torch.ones(n_tok_pad)
        aux["tok_pad_tt"] = ft(ones.reshape(1, n_tok_pad))
        aux["tok_col_pad_tt"] = ft(ones.reshape(1, n_tok_pad, 1))
    print(f"[{time.perf_counter()-t0:.0f}s] device aux built"
          f"{' (token mask forced to ones)' if a.mask_ones else ''}", flush=True)

    ttnn.from_torch = recording_from_torch
    try:
        with device_dtype_override(act):
            mod = OF3DiffusionModule(dmsd, cfg)
    finally:
        ttnn.from_torch = orig_from_torch
    walked = _device_weights(mod)
    for t in walked.values():
        ag.parameter(t)
    named = {id(t): reg.get(id(t)) for t in walked.values()}
    orient_by_name = {reg[i]: o for i, o in orient.items() if i in reg}
    n_named = sum(1 for v in named.values() if v)
    print(f"[{time.perf_counter()-t0:.0f}s] module built: {len(walked)} device weights "
          f"reachable, {n_named} carry a checkpoint name, {len(reg)} recorded at load",
          flush=True)

    si_trunk_d = ft(sq(kw["si_trunk"]).float().unsqueeze(0))
    zij_d = ft(zij_ref.reshape(1, n_token, n_token, -1).float())
    tok_pad = torch.zeros(n_tok_pad); tok_pad[:n_token] = token_mask
    xl_all = kw["xl_noisy"]
    t_all = kw["t"]

    which = list(range(N_STRUCT)) if a.structs == "all" else [int(x) for x in a.structs.split(",")]
    print(f"[{time.perf_counter()-t0:.0f}s] structures: {which[:5]}{'...' if len(which)>5 else ''}"
          f" ({len(which)} of {N_STRUCT})", flush=True)


    def _np(x):
        return ttnn.to_torch(x.value if hasattr(x, "value") else x).double()

    def _rel(x, y):
        x = x.reshape(-1)[: y.numel()] if x.numel() >= y.numel() else x.reshape(-1)
        y = y.reshape(-1)[: x.numel()]
        return float(torch.linalg.vector_norm(x - y)
                     / (torch.linalg.vector_norm(y) + 1e-300))

    def _pick(v, k):
        """Their intermediates carry the 48-sample axis; take structure k's slice."""
        if not torch.is_tensor(v):
            return None
        return v[0, k] if v.dim() >= 3 and v.shape[0] == 1 and v.shape[1] == N_STRUCT else \
            (v[0, 0] if v.dim() >= 3 and v.shape[0] == 1 and v.shape[1] == 1 else v)

    def bisect_stages(res, k):
        _, ai_glue, ai_dit, rl_upd, plm_pu, ql_enc = res
        dit_in = S["dit_in"][1]
        dec_in = S["dec_in"][1]
        out = {}
        try:
            out["ai_into_dit"] = _rel(_np(ai_glue), _pick(dit_in.get("a"), k))
        except Exception as e:
            out["ai_into_dit"] = f"ERR {e}"
        try:
            ours = _np(ai_dit)
            theirs = _pick(DIT_REF, k)
            out["ai_out_of_dit"] = _rel(ours, theirs)
            # Split by the token mask. This crop is 56 real tokens of 384, so a DiT that
            # leaks across pad rows shows up here and is invisible on a fixture at 79 %
            # occupancy. If the error sits in the pad rows the cause is masking; if it sits
            # in the real rows it is arithmetic and masking is exonerated.
            o = ours.reshape(-1, ours.shape[-1])[:theirs.shape[-2]]
            th = theirs.reshape(-1, theirs.shape[-1])
            m = token_mask.bool()[: o.shape[0]]
            for lbl, sel in (("real", m), ("pad", ~m)):
                if int(sel.sum()) == 0:
                    continue
                a_, b_ = o[sel], th[sel]
                out[f"ai_out_of_dit_{lbl}"] = float(
                    torch.linalg.vector_norm(a_ - b_)
                    / (torch.linalg.vector_norm(b_) + 1e-300))
                out[f"ai_out_of_dit_{lbl}_theirnorm"] = float(torch.linalg.vector_norm(b_))
                out[f"ai_out_of_dit_{lbl}_ournorm"] = float(torch.linalg.vector_norm(a_))
                out[f"ai_out_of_dit_{lbl}_rows"] = int(sel.sum())
        except Exception as e:
            out["ai_out_of_dit"] = f"ERR {e}"
        for nm, key in (("ql_encoder", "ql"), ("plm_encoder", "plm"), ("cl_encoder", "cl")):
            try:
                src = {"ql": ql_enc, "plm": plm_pu, "cl": None}[key]
                out[nm] = "n/a" if src is None else _rel(_np(src), _pick(dec_in.get(key), k))
            except Exception as e:
                out[nm] = f"ERR {e}"
        try:
            # their decoder output, reconstructed from the published xl_out
            sd2 = sigma_data ** 2
            tk_ = float(t_all[0, k])
            c_skip = sd2 / (sd2 + tk_ * tk_)
            c_out = sigma_data * tk_ / math.sqrt(sd2 + tk_ * tk_)
            their_rl = (S["xl_out"][0, k].double() - c_skip * xl_all[0, k].double()) / c_out
            out["rl_update_decoder"] = _rel(_np(rl_upd), their_rl)
        except Exception as e:
            out["rl_update_decoder"] = f"ERR {e}"
        return out

    DIT_REF = S["dit_out"]
    if a.mask_ones:
        DIT_REF = torch.load(f"{a.cap}/dit_out_maskones.pt", map_location="cpu",
                             weights_only=False)["dit_out_maskones"]
        token_mask = torch.ones_like(token_mask)
    err = None
    done = []
    probe = []
    fwd = []
    stages = {}
    for k in which:
        tk = float(t_all[0, k])
        xl_k = xl_all[0, k].float()
        rl_k = (xl_k * atom_mask[:, None]) / math.sqrt(tk * tk + sigma_data ** 2)
        rl_pad = torch.zeros(NP, 3); rl_pad[:n_atom] = rl_k
        si_k = torch.zeros(n_tok_pad, si_all.shape[-1])
        si_k[:n_token] = si_all[0, k].float()
        call = dict(
            si_trunk=ag.Tensor(si_trunk_d, requires_grad=True),
            si=ag.Tensor(ft(si_k.unsqueeze(0)), requires_grad=True),
            zij=ag.Tensor(zij_d, requires_grad=True),
            cl0=ag.Tensor(aux["cl0_d"], requires_grad=True),
            plm0=ag.Tensor(aux["plm0_d"], requires_grad=True),
            rl_noisy=ag.Tensor(ft(rl_pad.unsqueeze(0)), requires_grad=True),
            xl_noisy=ag.Tensor(ft(xl_k.unsqueeze(0)), requires_grad=True))
        cot_k = (which[(which.index(k) + 1) % len(which)]) if a.permute_cot else k
        seed = ft(cot[0, cot_k].float().unsqueeze(0))
        tk0 = time.perf_counter()
        try:
            with device_dtype_override(act), ag.tape():
                res = mod(call["si_trunk"], call["si"], call["zij"], call["cl0"], call["plm0"],
                          call["rl_noisy"], call["xl_noisy"],
                          aux["amc_d"], aux["amc_na_d"], aux["idx_tt"], aux["flat_tt"],
                          aux["zij_mask_d"], aux["kidx_tt"], aux["valid_d"], aux["mb_d"],
                          aux["pm_d"], aux["mean_d"], aux["tok_pad_tt"], aux["tok_col_pad_tt"],
                          n_atom, NP, nb, n_token, n_tok_pad, tk, sigma_data,
                          _return_intermediates=a.bisect)
                out = res[0] if a.bisect else res
                if a.bisect:
                    stages[k] = bisect_stages(res, k)
                ag.backward([out], [seed])
            fwd_here = ttnn.to_torch(out.value if hasattr(out, "value") else out).double()
            fwd_ref = S["xl_out"][0, k].double()
            fwd_here = fwd_here.reshape(fwd_ref.shape)
            fwd.append(float(torch.linalg.vector_norm(fwd_here - fwd_ref)
                             / (torch.linalg.vector_norm(fwd_ref) + 1e-300)))
            done.append(k)
            # Accumulation probe. The 48 structures are summed by running 48 tapes and letting
            # the leaves accumulate, which is exact only if `backward` adds into an existing
            # `.grad` across tape contexts rather than replacing it. A norm that grows structure
            # over structure is the evidence; one that stays flat means the run is measuring the
            # LAST structure alone and the total is wrong.
            pr = None
            for dt in walked.values():
                lf = ag._PARAMS.get(id(dt))
                if getattr(lf, "grad", None) is not None:
                    gv = lf.grad.value if hasattr(lf.grad, "value") else lf.grad
                    pr = float(ttnn.to_torch(gv).double().norm())
                    break
            probe.append(pr)
            print(f"[{time.perf_counter()-t0:.0f}s]   structure {k}: "
                  f"{time.perf_counter()-tk0:.1f}s, forward rel {fwd[-1]:.6e}, "
                  f"probe grad norm {pr}", flush=True)
        except Exception as e:
            err = e
            print(f"STRUCTURE {k} RAISED {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exception(type(e), e, e.__traceback__)
            break

    # ---- compare, transposing back by shape ---------------------------------------------------
    def to_t(x):
        return ttnn.to_torch(x).double()

    rows, missing = {}, []
    for dev_t in walked.values():
        nm = named.get(id(dev_t))
        if nm is None:
            continue
        leaf = ag._PARAMS.get(id(dev_t))
        g = getattr(leaf, "grad", None)
        if g is None:
            missing.append(nm)
            continue
        gt = to_t(g.value if hasattr(g, "value") else g)
        want = shape_by_name[nm]
        gt = gt.reshape(gt.shape[-len(want):]) if gt.dim() > len(want) else gt
        # D83: orientation from the recorded load, with the old shape test kept only for the
        # weights whose load was not a plain transpose ("?"). The shape test is a no-op on a
        # square weight, which is exactly the case it used to get wrong.
        o = orient_by_name.get(nm, "?")
        if o == "T" and gt.dim() == 2:
            gt = gt.t().contiguous()
        elif o != "N" and tuple(gt.shape) != want and tuple(gt.shape)[::-1] == want:
            gt = gt.t().contiguous()
        if tuple(gt.shape) != want:
            raise AssertionError(f"{nm}: gradient {tuple(gt.shape)} is not {want} after "
                                 f"orientation {o}")
        rows[nm] = gt

    if a.dump_grads:
        d = os.path.dirname(a.dump_grads)
        if d:
            os.makedirs(d, exist_ok=True)
        # Full checkpoint names, so the file drops straight into any comparison that uses the
        # upstream naming; `rows` is keyed relative to the `diffusion_module` sub-dict.
        torch.save({f"diffusion_module.{nm}": g.cpu() for nm, g in rows.items()}, a.dump_grads)
        print(f"[{time.perf_counter()-t0:.0f}s] wrote {len(rows)} device gradient tensors to "
              f"{a.dump_grads}", flush=True)

    tot_ref_sq = sum(float(v.double().pow(2).sum()) for v in ref_grad.values() if v is not None)
    cmp_rows, worst, worst_n = [], -1.0, None
    sq_cmp = 0.0
    per_tensor = []
    for nm, gt in rows.items():
        r = ref_grad.get(nm)
        if r is None or tuple(gt.shape) != tuple(r.shape):
            continue
        rd = r.double()
        rn = float(torch.linalg.vector_norm(rd))
        d = float(torch.linalg.vector_norm(gt - rd) / (rn + 1e-300))
        cmp_rows.append((d, nm, rn))
        sq_cmp += rn * rn
        if d > worst:
            worst, worst_n = d, nm
        # PROTOCOL A16/D35. This instrument only ever emitted `rel`, and a single rel cannot say
        # WHICH WAY we are wrong: rel^2 = 1 + r^2 - 2rc bounds the norm ratio to [1-rel, 1+rel]
        # only, and a zero gradient reads r = 0, rel = 1. The trunk arms have carried norm_ratio
        # and cos for several passes; the diffusion arm did not, which is why the DiT could not
        # be read the way the trunk ladder was. Two more floats, already in memory, free.
        dn = float(torch.linalg.vector_norm(gt))
        per_tensor.append({
            "tensor": nm, "rel_l2": d, "ref_norm": rn, "device_norm": dn,
            "norm_ratio": (dn / rn) if rn else None,
            "cos": (float((gt * rd).sum() / (dn * rn)) if (dn and rn) else None),
        })
    cmp_rows.sort()
    med = cmp_rows[len(cmp_rows) // 2][0] if cmp_rows else None

    # zero-model baseline, MEASURED beside the median (A16) rather than assumed to be 1.0
    zero = sorted(1.0 for _ in cmp_rows)
    zmed = zero[len(zero) // 2] if zero else None

    rep = {"structures_done": done, "accumulation_probe": probe, "bisect": stages, "forward_rel": fwd,
           "forward_rel_median": (sorted(fwd)[len(fwd)//2] if fwd else None), "structures_asked": which, "n_struct_total": N_STRUCT,
           "tokens": n_token, "atoms": n_atom, "nb": nb, "NP": NP,
           "device_weights_reachable": len(walked), "device_weights_named": n_named,
           "weights_with_grad": len(rows), "weights_without_grad": len(missing),
           "compared": len(cmp_rows),
           "reference_tensors": len(ref_grad),
           "share_of_diffusion_sq_norm": sq_cmp / tot_ref_sq if tot_ref_sq else None,
           "median_rel": med, "worst_rel": worst, "worst_tensor": worst_n,
           "over_5e-2": sum(1 for d, _, _ in cmp_rows if d > 5.0e-2),
           "zero_model_median": zmed,
           "per_tensor": (per_tensor if a.dump_per_tensor else None),
           "orientation_recorded_at_load": {
               "N": sum(1 for o in orient_by_name.values() if o == "N"),
               "T": sum(1 for o in orient_by_name.values() if o == "T"),
               "?": sum(1 for o in orient_by_name.values() if o == "?")},
           "square_2d_compared": sum(1 for nm in rows
                                     if len(shape_by_name[nm]) == 2
                                     and shape_by_name[nm][0] == shape_by_name[nm][1]),
           "softmax_f64_bound": bool(a.softmax_f64),
           "softmax_lever": a.softmax_lever or None,
           "chain_sum_ckc": ("none" if os.environ.get("OF3T_NANFLOOR_SUM_CKC") == "none"
                             else "precise") if a.softmax_lever == "accurate" else None,
           "chain_sum_ckc_census": (sum_ckc_census if a.softmax_lever == "accurate" else None),
           "ckc_census": (ckc_seen or None),
           "tt_bio_softmax_ckc_flag": (
               __import__("tt_bio.tenstorrent", fromlist=["x"])._SOFTMAX_CKC
               if a.ckc_census else None),
           "fp32_softmax_stats": (
               dict(__import__("tt_bio.tenstorrent", fromlist=["x"]).FP32_SOFTMAX_STATS)
               if a.ckc_census else None),
           "nonfinite_probe": (bad[:40] if a.softmax_lever == "accurate" else None),
           "nonfinite_probe_count": (len(bad) if a.softmax_lever == "accurate" else None),
           "probe2_len": len(trace) or None,
           "probe3_len": len(fwd_trace) or None,
           "probe3_first_bad": next(
               ({"index": i, "verb": v, "shape": list(s),
                 "previous": [{"verb": pv, "finite": pok, "shape": list(ps)}
                              for pv, pok, ps in fwd_trace[max(0, i - 8):i]]}
                for i, (v, ok, s) in enumerate(fwd_trace) if not ok),
               ("ALL FINITE" if fwd_trace else None)),
           "probe2_first_bad": next(
               ({"index": i, "verb": v, "shape": list(s),
                 "previous": [{"verb": pv, "finite": pok, "shape": list(ps)}
                              for pv, pok, ps in trace[max(0, i - 8):i]]}
                for i, (v, ok, s) in enumerate(trace) if not ok),
               ("ALL FINITE" if trace else None)),
           "softmax_calls_intercepted": softmax_calls[0],
           "softmax_site_f64": bool(a.softmax_site_f64),
           "host_f64_softmax_stats": dict(T.HOST_F64_SOFTMAX_STATS),
           "host_f64_softmax_ab": os.environ.get("TT_BIO_HOST_F64_SOFTMAX_AB") or None,
           "host_f64_verbs": hf64_verbs or None,
           "host_f64_calls_intercepted": (
               {v: __import__("host_f64").CALLS.get(v, 0) for v in hf64_verbs}
               if hf64_verbs else None),
           "verb_census": (dict(sorted(__import__("host_f64").CALLS.items()))
                           if a.verb_census else None),
           "cotangent_permuted": bool(a.permute_cot),
           "grads_dumped_to": a.dump_grads or None,
           "per_tensor_dumped": bool(a.dump_per_tensor),
           # PROVENANCE. This instrument scores against a captured boundary and against a
           # checkpoint, and until now recorded neither in its output. `device_gradient_043pt`
           # had to be tied back to `device_gradient_043all` by showing 48 forward_rel doubles
           # were bit-identical, because no artifact said which `--cap` either one read. An
           # artifact's identity is its digest plus its recorded inputs.
           "provenance": {"cap": a.cap, "ckpt": CKPT, "out_dir": a.out_dir, "tag": a.tag,
                          "structs": a.structs, "mask_ones": bool(a.mask_ones),
                          "argv": sys.argv},
           "best10": [(n, d) for d, n, _ in cmp_rows[:10]],
           "worst10": [(n, d) for d, n, _ in cmp_rows[-10:]],
           "error": None if err is None else f"{type(err).__name__}: {err}"}
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"device_gradient{a.tag}.json")
    # The sidecar is UNCONDITIONAL, and that is the whole point of it. A result file that keeps
    # only aggregates and a worst/best 10 cannot be re-analysed under a denominator discovered
    # later, and this campaign has changed its denominator twice: A15 moved reach from a count
    # of tensors to a share of the squared norm, and A23 moved the headline from a median over
    # tensors to a mass-weighted figure. Both re-scorings need the array, neither was
    # foreseeable when the run was taken, and the array costs a few hundred kB.
    side = os.path.join(a.out_dir, f"device_gradient{a.tag}_per_tensor.json")
    json.dump({"per_tensor": per_tensor, "compared": len(cmp_rows),
               "provenance": rep["provenance"],
               "model_sq_norm_043": 10.279642678524985,
               "what": "every compared tensor with rel_l2, ref_norm, device_norm, norm_ratio "
                       "and cos, so any later denominator can be applied without a card run"},
              open(side, "w"), indent=1, sort_keys=True, default=str)
    json.dump(rep, open(path, "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("best10", "worst10", "per_tensor")},
                     indent=1, default=str), flush=True)
    print("wrote", path, "and", side, flush=True)
    if hf64_verbs:
        counts, zero = __import__("host_f64").report(hf64_verbs)
        print("host_f64 intercepts:", counts, flush=True)
        if zero:
            print(f"FAILED: --host-f64 intercepted 0 calls for {zero}, so those verbs are the "
                  f"shipped ones under another name", flush=True)
            return 3
    if a.ckc_census and softmax_calls[0] == 0:
        print("FAILED: --ckc-census saw 0 softmax calls, so the census is empty rather than "
              "informative", flush=True)
        return 3
    if a.softmax_site_f64 and T.HOST_F64_SOFTMAX_STATS["served"] == 0:
        print("FAILED: --softmax-site-f64 served 0 softmax calls, so this arm is the shipped arm "
              "under another name", flush=True)
        return 3
    if (a.softmax_f64 or a.softmax_lever) and softmax_calls[0] == 0:
        print(f"FAILED: {'--softmax-f64' if a.softmax_f64 else '--softmax-lever'} intercepted "
              f"0 softmax calls, so this arm is the shipped arm under another name", flush=True)
        return 3
    return 1 if err is not None else 0


if __name__ == "__main__":
    sys.exit(main())

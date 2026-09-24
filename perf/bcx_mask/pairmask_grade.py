"""What does the MISSING PAIR MASK cost, at the state BindCraft 2's trajectories actually ran?

`bcx-predictor` landed three mask sites -- the two MSA attentions and the outer product mean --
and every one of them reads `msa_mask`. The pair track reads nothing: `AF2EvoformerBlock.__call__`
forwards `mask=None, attn_mask=None` because `splice.py:95` never passes a pair mask, so at
`length_bucket_size` 32 the two triangle multiplications and the two triangle attentions run
unmasked over 19 padded residues of 211.

Four arms of identical torch code on BindCraft 2's own captured Evoformer input, fp32, scored
against BindCraft 2's own JAX output of the same 48 blocks. Precision cancels between arms and
only the masking differs.

  af2            AF2's masking: `mask * p_in(x)` (BOTH halves) and the `1e9*(mask-1)` key bias
  device_today   pair mask nowhere in the pair track -- what the device computes now
  trimul_only    pair mask in the two multiplications, no key bias -- locates the carrier
  device_fixed   pair mask in both, but the multiplication masks ONE half, which is what
                 `tenstorrent.py:7326` does and what this row's fix would ship

`device_fixed` against `af2` is the whole both-halves question, and it is measured rather than
argued. Every distance is taken over the REAL residues only, selected by the mask: at bucket 32
the 19 pad residues sit at 77..95, between the binder and the target, so a prefix slice is wrong.

Run at bucket 32 (n=211, 19 masked) and again at bucket 1 (n=192, unmasked) as the control:
a single masked distance says nothing, the difference between the two is the defect's size.
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "perf" / "shared"), str(ROOT)):
    sys.path.insert(0, p)

import jax                                                             # noqa: E402
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

import afgrad as A                                                     # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af.alphafold.model import modules                       # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402

PARAM_DIR = os.environ.get("BCX_PARAM_DIR", "/home/ttuser/bcx_e2e/af2_params")
PARAM_NPZ = os.path.join(PARAM_DIR, "params_model_1_ptm.npz")


def find_evoformer_masks(fn, depth=0, seen=None):
    """`splice.py:193`'s closure walk, imported by copy so this file runs standalone."""
    import types
    seen = seen if seen is not None else set()
    if depth > 6 or not isinstance(fn, types.FunctionType) or id(fn) in seen:
        return None
    seen.add(id(fn))
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if name == "evoformer_masks" and isinstance(value, dict) and "msa" in value:
            return value
        found = find_evoformer_masks(value, depth + 1, seen)
        if found is not None:
            return found
    return None


def capture(bucket: int) -> dict:
    """BindCraft 2's own Evoformer stack: its input, its masks and its output, first recycle.

    One callback holding all six arrays, so the input and the output can never come from
    different recycles.
    """
    from bindcraft.af.alphafold.model import layer_stack as LS
    got: dict = {}

    def store(mi, pi, mm, pm, mo, po):
        if got:
            return
        got.update(msa_in=np.asarray(mi, np.float32), pair_in=np.asarray(pi, np.float32),
                   msa_mask=np.asarray(mm, np.float32), pair_mask=np.asarray(pm, np.float32),
                   msa_out=np.asarray(mo, np.float32), pair_out=np.asarray(po, np.float32))

    real = LS.layer_stack

    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)

        def choose(fn):
            built = made(fn)
            if getattr(fn, "__name__", None) != "evoformer_fn":
                return built
            masks = find_evoformer_masks(fn)
            assert masks is not None and "pair" in masks, "no evoformer_masks in the closure"

            def watched(x):
                act, safe_key = x
                out, key_out = built(x)
                jax.debug.callback(store, act["msa"], act["pair"], masks["msa"], masks["pair"],
                                   out["msa"], out["pair"])
                return out, key_out
            return watched
        return choose

    modules.layer_stack.layer_stack = factory
    try:
        settings = B.campaign_settings(overrides=[f"length_bucket_size={bucket}"])
        _, states, _ = B.design_state(settings)
        TTBioAlphaFoldDesignModel(
            presets=("model_1_ptm",), data_dir=PARAM_DIR, models=("model_1_ptm",), num_recycle=1,
            key=jax.random.PRNGKey(0), length_bucket_size=bucket, max_cache_size=2,
            trunk="jax").predict(states)
    finally:
        modules.layer_stack.layer_stack = real
    assert got, "the Evoformer stack never ran"
    return got


def one_sided_trimul():
    """The reference's triangle multiplication with only the `a` half masked.

    `tt_bio/af2_reference.py:242` is `mask * p_in(x)` over the fused 2*hidden projection, which
    is AF2's both halves. `tt_bio/tenstorrent.py:7326` multiplies `a_chunk` alone, and its own
    comment at :7155 says so. This is that one, in torch.
    """
    import tt_bio.af2_reference as R

    def forward(self, z, mask):
        x = self.norm_in(z)
        proj = self.p_in(x)
        a, b = proj.split(self.hidden, dim=-1)
        a = mask.unsqueeze(-1).to(x.dtype) * a
        proj = torch.cat([a, b], dim=-1) * torch.sigmoid(self.g_in(x))
        a, b = proj.split(self.hidden, dim=-1)
        equation = "kic,kjc->ijc" if self.ending else "ikc,jkc->ijc"
        act = self.p_out(self.norm_out(torch.einsum(equation, a, b)))
        return act * torch.sigmoid(self.g_out(x))
    return forward


def run_arms(cap: dict, ref) -> dict:
    import tt_bio.af2_reference as R
    both = R.TriangleMultiplication.forward
    one = one_sided_trimul()

    m0 = torch.from_numpy(cap["msa_in"]).float()
    z0 = torch.from_numpy(cap["pair_in"]).float()
    mm = torch.from_numpy(cap["msa_mask"]).float()
    pm = torch.from_numpy(cap["pair_mask"]).float()
    ones = torch.ones_like(pm)
    model = ref["f32"]

    def stack(pair_mask_mul, pair_mask_att, sided):
        R.TriangleMultiplication.forward = sided
        try:
            m, z = m0.clone(), z0.clone()
            with torch.no_grad():
                for i in range(48):
                    blk = model.evoformer[i]
                    m = blk._msa_track(m, z, mm)
                    z = z + blk.opm(m, mm)
                    z = z + blk.tri_mul_out(z, pair_mask_mul)
                    z = z + blk.tri_mul_in(z, pair_mask_mul)
                    z = z + blk.tri_att_start(z, pair_mask_att)
                    z = z + blk.tri_att_end(z, pair_mask_att)
                    z = z + blk.pair_transition(z)
            return m, z
        finally:
            R.TriangleMultiplication.forward = both

    return {"af2": stack(pm, pm, both),
            "device_today": stack(ones, ones, both),
            "trimul_only": stack(pm, ones, one),
            "device_fixed": stack(pm, pm, one)}


def grade(cap: dict, arms: dict, ref) -> dict:
    """Every arm against BindCraft 2's JAX, on the real residues the mask names."""
    keep = torch.from_numpy(cap["msa_mask"][0]).bool()          # row 0 is the sequence mask
    jm = torch.from_numpy(cap["msa_out"]).float()
    jz = torch.from_numpy(cap["pair_out"]).float()
    single = ref["f32"].single_activations

    def cmp(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        return {"rel_l2": float((a - b).norm() / b.norm()), "cos": float(a @ b / (a.norm() * b.norm()))}

    with torch.no_grad():
        js = single(jm[0])
        out = {}
        for name, (m, z) in arms.items():
            s = single(m[0])
            out[name] = {
                "msa": cmp(m[:, keep], jm[:, keep]),
                "pair": cmp(z[keep][:, keep], jz[keep][:, keep]),
                "single": cmp(s[keep], js[keep]),
            }
    return out


def main():
    _, ref = A.load_models(PARAM_NPZ, device_arm=False)
    out = {}
    for bucket in (32, 1):
        cap = capture(bucket)
        n = int(cap["pair_in"].shape[0])
        masked = int((cap["msa_mask"][0] == 0).sum())
        arms = run_arms(cap, ref)
        out[f"bucket_{bucket}"] = {
            "n": n, "n_masked": masked,
            "masked_at": np.flatnonzero(cap["msa_mask"][0] == 0).tolist(),
            "pair_mask_is_outer_product": bool(np.allclose(
                cap["pair_mask"], cap["msa_mask"][0][:, None] * cap["msa_mask"][0][None, :])),
            "arms": grade(cap, arms, ref),
        }
        print(json.dumps({f"bucket_{bucket}": out[f"bucket_{bucket}"]}, indent=1))
    (HERE / "pairmask_grade.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

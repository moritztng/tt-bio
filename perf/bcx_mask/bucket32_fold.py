"""Does a fold at the lab's DEFAULT bucket match its unpadded equivalent, on the fold's OUTPUT?

Op distances are not the answer to this question: the pair track feeds the structure module and
the structure module feeds pLDDT, so the number that decides it is pLDDT and the coordinates.

BindCraft 2's `length_bucket_size` defaults to 32 and every upstream user gets that default, so
the Evoformer runs over a padded, masked complex on every trajectory. Four arms, each a complete
BindCraft 2 `predict` of the same PD-L1 design state:

  jax_b1          BindCraft 2 unmodified, bucket 1  -- the unpadded truth
  jax_b32         BindCraft 2 unmodified, bucket 32 -- what AF2 itself does when padded
  host_today_b32  tt-bio's Evoformer masking as it is TODAY (msa mask only), bucket 32
  host_fixed_b32  tt-bio's Evoformer masking with the pair mask in, bucket 32

The two `host_*` arms run tt-bio's own 48 blocks through `af2_reference` in fp32 on the host,
spliced into BindCraft 2's haiku model by `jax.pure_callback`. Host and not device because this
row holds no card: the arithmetic is tt-bio's, the precision is not bf16, so a difference here
is a MASKING difference and the bf16 floor is a separate, already-measured question.

pLDDT is reported over the real residues only, which the mask names; the pad sits inside the
complex (the design chain is padded before the target) as well as at the end, so a prefix slice
is wrong. Coordinates are compared as CA RMSD after a Kabsch alignment of the real residues.
"""
import contextlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "perf" / "shared"), str(ROOT)):
    sys.path.insert(0, p)

import jax                                                             # noqa: E402
import jax.numpy as jnp                                                # noqa: E402
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

import afgrad as A                                                      # noqa: E402
import bc2_state as B                                                   # noqa: E402
from bindcraft.af.alphafold.model import modules                        # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                   # noqa: E402

from pairmask_grade import PARAM_DIR, PARAM_NPZ, find_evoformer_masks   # noqa: E402

CA = 1  # `residue_constants.atom_order['CA']`


class HostEvoformer:
    """tt-bio's 48 Evoformer blocks in host torch, as a JAX callback.

    The same cut `splice.py` makes -- `(msa, pair) -> (msa, pair)` around `evoformer_fn` -- with
    the reference stack in place of the card and forward only, because `predict` never
    differentiates. `mode` selects which masks the pair track gets:

      today  none, which is what `AF2EvoformerBlock` computes when its caller passes no pair
             mask (`splice.py:95` passes `masks["msa"]` and nothing else)
      fixed  the pair mask in the two multiplications, one-sided as `tenstorrent.py:7326`
             masks, and the key bias in the two attentions, broadcast over the query axis as
             `af2.af2_pair_masks` builds it
      af2    AF2's own masking, both halves and the per-row bias -- the upper bound on `fixed`
    """

    def __init__(self, ref, mode: str):
        self.model, self.mode = ref["f32"], mode
        self.calls = 0

    def _stack(self, msa_np, pair_np, msa_mask_np, pair_mask_np):
        import tt_bio.af2_reference as R
        from pairmask_grade import one_sided_trimul
        both = R.TriangleMultiplication.forward
        sided = both if self.mode == "af2" else one_sided_trimul()
        m = torch.from_numpy(np.asarray(msa_np).copy()).float()
        z = torch.from_numpy(np.asarray(pair_np).copy()).float()
        mm = torch.from_numpy(np.asarray(msa_mask_np).copy()).float()
        pm = torch.from_numpy(np.asarray(pair_mask_np).copy()).float()
        if self.mode == "today":
            pm_mul = pm_att = torch.ones_like(pm)
        elif self.mode == "fixed":
            # The broadcast key bias: every query row gets `1e9 * (seq_mask - 1)`.
            seq = torch.diagonal(pm)
            pm_mul, pm_att = pm, torch.ones_like(pm) * seq[None, :]
        else:
            pm_mul = pm_att = pm
        R.TriangleMultiplication.forward = sided
        try:
            with torch.no_grad():
                for i in range(48):
                    blk = self.model.evoformer[i]
                    m = blk._msa_track(m, z, mm)
                    z = z + blk.opm(m, mm)
                    z = z + blk.tri_mul_out(z, pm_mul)
                    z = z + blk.tri_mul_in(z, pm_mul)
                    z = z + blk.tri_att_start(z, pm_att)
                    z = z + blk.tri_att_end(z, pm_att)
                    z = z + blk.pair_transition(z)
        finally:
            R.TriangleMultiplication.forward = both
        self.calls += 1
        return m.numpy().astype(np.float32), z.numpy().astype(np.float32)

    @contextlib.contextmanager
    def installed(self, expect_blocks: int = 48):
        """`splice.evoformer_on_device`'s swap, with the PAIR mask passed as well as the MSA one."""
        from bindcraft.af.alphafold.model import layer_stack as LS
        real = LS.layer_stack

        def call(msa, pair, msa_mask, pair_mask):
            out = jax.pure_callback(
                self._stack,
                (jax.ShapeDtypeStruct(msa.shape, jnp.float32),
                 jax.ShapeDtypeStruct(pair.shape, jnp.float32)),
                msa.astype(jnp.float32), pair.astype(jnp.float32),
                msa_mask.astype(jnp.float32), pair_mask.astype(jnp.float32))
            return out[0].astype(msa.dtype), out[1].astype(pair.dtype)

        def factory(num_layers, *a, **kw):
            made = real(num_layers, *a, **kw)

            def choose(fn):
                if getattr(fn, "__name__", None) != "evoformer_fn":
                    return made(fn)
                assert int(num_layers) == expect_blocks, num_layers
                masks = find_evoformer_masks(fn)
                assert masks is not None and "pair" in masks

                def on_host(x):
                    act, safe_key = x
                    msa, pair = call(act["msa"], act["pair"], masks["msa"], masks["pair"])
                    return {**act, "msa": msa, "pair": pair}, safe_key
                return on_host
            return choose

        modules.layer_stack.layer_stack = factory
        try:
            yield
        finally:
            modules.layer_stack.layer_stack = real


def fold(bucket: int, host: HostEvoformer | None) -> dict:
    """One BindCraft 2 `predict`, and what it returns about the REAL residues only.

    `af2.py:323-326` trims the prediction back to the true chain lengths and
    `trim_padded_metrics` trims the metrics with it, so a padded arm and an unpadded arm come
    back on the same residues by BindCraft 2's own accounting and nothing here has to guess
    where the pad was.
    """
    settings = B.campaign_settings(overrides=[f"length_bucket_size={bucket}"])
    _, states, _ = B.design_state(settings)
    model = TTBioAlphaFoldDesignModel(
        presets=("model_1_ptm",), data_dir=PARAM_DIR, models=("model_1_ptm",), num_recycle=1,
        key=jax.random.PRNGKey(0), length_bucket_size=bucket, max_cache_size=2, trunk="jax")
    ctx = host.installed() if host is not None else contextlib.nullcontext()
    with ctx:
        out = model.predict(states)
    name = sorted(out)[0]
    pred = out[name]
    chains = sorted(pred.protein_complex)
    ca = np.concatenate([np.asarray(pred.protein_complex[c].atoms, np.float64)[:, CA, :]
                         for c in chains], axis=0)
    return {"state": name, "chains": chains,
            "chain_lengths": [len(pred.protein_complex[c]) for c in chains],
            "plddt": np.asarray(pred.metrics["plddt"], np.float64), "ca": ca,
            "calls": None if host is None else host.calls}


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))


def main():
    _, ref = A.load_models(PARAM_NPZ, device_arm=False)
    arms = {"jax_b1": (1, None), "jax_b32": (32, None),
            "host_today_b32": (32, HostEvoformer(ref, "today")),
            "host_fixed_b32": (32, HostEvoformer(ref, "fixed"))}
    only = os.environ.get("BCX_ARMS")
    if only:
        arms = {k: arms[k] for k in only.split(",")}
    got, out = {}, {"arms": {}}
    for name, (bucket, host) in arms.items():
        r = got[name] = fold(bucket, host)
        print(name, "chains", r["chain_lengths"], "plddt n", int(r["plddt"].shape[0]),
              "mean", round(float(r["plddt"].mean()), 4), flush=True)
        out["arms"][name] = {"chain_lengths": r["chain_lengths"],
                             "plddt_n": int(r["plddt"].shape[0]),
                             "mean_plddt": float(r["plddt"].mean()),
                             "stack_calls": r["calls"],
                             "plddt": [round(float(v), 5) for v in r["plddt"]]}
        (HERE / "bucket32_fold.json").write_text(json.dumps(out, indent=1))

    base = got.get("jax_b1")
    if base is not None:
        for name, r in got.items():
            if name == "jax_b1" or r["plddt"].shape != base["plddt"].shape:
                out["arms"][name]["vs_jax_b1"] = (
                    None if name == "jax_b1" else f"shape {r['plddt'].shape} vs {base['plddt'].shape}")
                continue
            out["arms"][name]["vs_jax_b1"] = {
                "mean_plddt_delta": float(r["plddt"].mean() - base["plddt"].mean()),
                "max_abs_plddt_delta": float(np.abs(r["plddt"] - base["plddt"]).max()),
                "ca_rmsd_kabsch_A": kabsch_rmsd(r["ca"], base["ca"]),
            }
    (HERE / "bucket32_fold.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "plddt"}
                      for k, v in out["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()

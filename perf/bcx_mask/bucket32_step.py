"""Does a design STEP at the lab's default bucket match its unpadded equivalent, on its outputs?

The program every BindCraft 2 design step runs is `sequence_gradients`, and it is the one that
pads: `bindcraft/af2.py:385` pads the design chain to `length_bucket_size`, so at the default 32
PD-L1 runs as 96 + 115 = 211 with 19 masked residues in the middle of the complex, and at 1 it
runs unpadded at 192. AF2's own masking makes the two the same computation on the real residues,
so the step's outputs -- `design_loss`, pLDDT, and the sequence gradient the optimiser follows --
must agree between the buckets. Whether they do through tt-bio's masking is the question, and
whether any of them is non-finite is the question `trajectory.py:134-136` asks every step.

Arms, one step each, same state, dropout off in every arm (the replaced stack has none):

  jax_b1           BindCraft 2 unmodified, bucket 1  -- the unpadded truth
  jax_b32          BindCraft 2 unmodified, bucket 32 -- AF2's own masking, padded
  host_b1          tt-bio's 48 blocks, bucket 1      -- the host stack's floor against JAX
  host_today_b32   tt-bio's masking as the device has it today: MSA mask only
  host_fixed_b32   tt-bio's masking with `af2_pair_masks`: one-sided multiply, key bias

The host arms run `finite_probe.run_stack` -- the reference's blocks in fp32 on the host with the
arm's masks -- spliced into BindCraft 2's haiku model as a `jax.custom_vjp` whose backward is
torch autograd over the same 48 blocks. Host and not device because this row holds no card: a
difference here is a MASKING difference, and bf16 is a separate, already-measured question.
"""
import contextlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import finite_probe as P                                                # noqa: E402
import pairmask_grade as G                                              # noqa: E402

import jax                                                              # noqa: E402
import jax.numpy as jnp                                                 # noqa: E402
import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

import bc2_state as B                                                   # noqa: E402
from bindcraft.af.alphafold.model import modules                        # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                   # noqa: E402

OUT = HERE / "bucket32_step.json"


class HostEvoformer:
    """`(msa, pair, msa_mask, pair_mask) -> (msa, pair)` in host torch, differentiable in JAX."""

    def __init__(self, model, mode: str):
        self.model, self.mode = model, mode
        self.calls = {"primal": 0, "vjp": 0}

    @staticmethod
    def _t(x):
        return torch.from_numpy(np.asarray(x, np.float32).copy())

    def _primal(self, m, z, mm, pm):
        with torch.no_grad():
            mo, zo = P.run_stack(self.model, self._t(m), self._t(z), self._t(mm), self._t(pm),
                                 self.mode)
        self.calls["primal"] += 1
        return mo.numpy(), zo.numpy()

    def _vjp(self, m, z, mm, pm, gm, gz):
        one = P.arm_masks(self.mode, self._t(pm))[3]
        ml = self._t(m).requires_grad_(True)
        zl = self._t(z).requires_grad_(True)
        with P.trimul_mode(one), torch.enable_grad():
            mo, zo = P.run_stack(self.model, ml, zl, self._t(mm), self._t(pm), self.mode,
                                 hold=True)
            ((mo * self._t(gm)).sum() + (zo * self._t(gz)).sum()).backward()
        self.calls["vjp"] += 1
        return ml.grad.numpy(), zl.grad.numpy()

    def as_jax(self):
        f32 = lambda *xs: tuple(jax.ShapeDtypeStruct(x.shape, jnp.float32) for x in xs)

        @jax.custom_vjp
        def stack(m, z, mm, pm):
            mo, zo = jax.pure_callback(self._primal, f32(m, z), m.astype(jnp.float32),
                                       z.astype(jnp.float32), mm.astype(jnp.float32),
                                       pm.astype(jnp.float32))
            return mo.astype(m.dtype), zo.astype(z.dtype)

        def fwd(m, z, mm, pm):
            return stack(m, z, mm, pm), (m, z, mm, pm)

        def bwd(res, cts):
            m, z, mm, pm = res
            gm, gz = jax.pure_callback(self._vjp, f32(m, z), m.astype(jnp.float32),
                                       z.astype(jnp.float32), mm.astype(jnp.float32),
                                       pm.astype(jnp.float32), cts[0].astype(jnp.float32),
                                       cts[1].astype(jnp.float32))
            return gm.astype(m.dtype), gz.astype(z.dtype), jnp.zeros_like(mm), jnp.zeros_like(pm)

        stack.defvjp(fwd, bwd)
        return stack

    @contextlib.contextmanager
    def installed(self):
        """`splice.evoformer_on_device`'s swap, passing the PAIR mask as well as the MSA one."""
        from bindcraft.af.alphafold.model import layer_stack as LS
        real, host = LS.layer_stack, self.as_jax()

        def factory(num_layers, *a, **kw):
            made = real(num_layers, *a, **kw)

            def choose(fn):
                if getattr(fn, "__name__", None) != "evoformer_fn":
                    return made(fn)
                assert int(num_layers) == 48, num_layers
                masks = G.find_evoformer_masks(fn)
                assert masks is not None and "pair" in masks

                def on_host(x):
                    act, key = x
                    m, z = host(act["msa"], act["pair"], masks["msa"], masks["pair"])
                    return {**act, "msa": m, "pair": z}, key
                return on_host
            return choose

        modules.layer_stack.layer_stack = factory
        try:
            yield
        finally:
            modules.layer_stack.layer_stack = real


def step(bucket: int, host: HostEvoformer | None) -> dict:
    settings = B.campaign_settings(overrides=[f"length_bucket_size={bucket}"])
    _, states, losses = B.design_state(settings)
    model = TTBioAlphaFoldDesignModel(
        presets=("model_1_ptm",), data_dir=G.PARAM_DIR, models=("model_1_ptm",), num_recycle=1,
        key=jax.random.PRNGKey(0), length_bucket_size=bucket, max_cache_size=2, dropout=False,
        trunk="jax")
    real_len = {name: len(p) for c in states.values() for name, p in c.items()}
    with (host.installed() if host is not None else contextlib.nullcontext()):
        predictions, grads, loss = model.sequence_gradients(states, losses)
    pred = predictions[sorted(predictions)[0]]
    metrics = {k: np.asarray(v, np.float64) for k, v in pred.metrics.items()}
    # The gradient path returns metrics on the PADDED complex. Its own flags say which residues
    # are real, so trim by them rather than by position.
    flags = np.concatenate([np.asarray(pred.protein_complex[c].flags)
                            for c in sorted(pred.protein_complex)])
    from bindcraft.protein import real_residue_weights
    real = np.asarray(real_residue_weights(flags)) > 0
    plddt = metrics["plddt"]
    g = {name: np.asarray(v, np.float64)[:real_len.get(name, len(v))] for name, v in grads.items()}
    return {"loss": float(loss), "loss_finite": bool(np.isfinite(float(loss))),
            "n": int(plddt.shape[0]), "n_real": int(real.sum()),
            "plddt_real": plddt[real] if plddt.shape[0] == real.shape[0] else plddt,
            "scalars": {k: float(v) for k, v in metrics.items() if v.ndim == 0},
            "grads": g,
            "grads_nonfinite": {k: int((~np.isfinite(v)).sum()) for k, v in g.items()},
            "calls": None if host is None else dict(host.calls)}


def compare(r, base):
    out = {"loss_delta": r["loss"] - base["loss"],
           "mean_plddt_delta": float(np.mean(r["plddt_real"]) - np.mean(base["plddt_real"])),
           "max_abs_plddt_delta": float(np.abs(r["plddt_real"] - base["plddt_real"]).max())
           if r["plddt_real"].shape == base["plddt_real"].shape else None,
           "grad": {}}
    for k, v in r["grads"].items():
        b = base["grads"].get(k)
        if b is None or b.shape != v.shape:
            continue
        out["grad"][k] = {
            "cos": float(v.ravel() @ b.ravel() / (np.linalg.norm(v) * np.linalg.norm(b))),
            "norm_ratio": float(np.linalg.norm(v) / np.linalg.norm(b)),
            "rel_l2": float(np.linalg.norm(v - b) / np.linalg.norm(b))}
    return out


def main():
    import afgrad as A
    _, ref = A.load_models(G.PARAM_NPZ, device_arm=False)
    model = ref["f32"]
    # The two arms the verdict rests on first, then the host floor.
    arms = {"jax_b1": (1, None), "jax_b32": (32, None),
            "host_today_b32": (32, HostEvoformer(model, "device_today")),
            "host_fixed_b32": (32, HostEvoformer(model, "device_fixed")),
            "host_b1": (1, HostEvoformer(model, "af2")),
            "host_af2_b32": (32, HostEvoformer(model, "af2"))}
    out = json.loads(OUT.read_text()) if OUT.is_file() else {"arms": {}}
    got = {}
    for name, (bucket, host) in arms.items():
        t0 = time.time()
        saved = HERE / f"step_{name}.npz"
        if name in out["arms"] and saved.is_file() and "loss" in out["arms"][name]:
            # A finished arm is read back, not rerun: one JAX step here is minutes of CPU.
            with np.load(saved) as z:
                got[name] = {"loss": out["arms"][name]["loss"], "plddt_real": z["plddt_real"],
                             "grads": {k[5:]: z[k] for k in z.files if k.startswith("grad_")}}
            print(name, "reused", flush=True)
            continue
        r = got[name] = step(bucket, host)
        # Every arm's own arrays on disk, so a comparison can be redone without a rerun.
        np.savez(HERE / f"step_{name}.npz", plddt_real=r["plddt_real"],
                 **{f"grad_{k}": v for k, v in r["grads"].items()})
        row = {"bucket": bucket, "seconds": round(time.time() - t0),
               "loss": r["loss"], "loss_finite": r["loss_finite"], "n": r["n"],
               "n_real": r["n_real"], "mean_plddt_real": float(np.mean(r["plddt_real"])),
               "plddt_real_nonfinite": int((~np.isfinite(r["plddt_real"])).sum()),
               "scalars": r["scalars"], "grads_nonfinite": r["grads_nonfinite"],
               "grad_norms": {k: float(np.linalg.norm(v)) for k, v in r["grads"].items()},
               "calls": r["calls"]}
        # Against the unpadded truth AND against JAX at the SAME bucket: BindCraft 2's own two
        # buckets are not the same program (`padded_to` continues `residue_index` through the
        # pad, so the monomer chain break moves), and only the same-bucket pair isolates masking.
        for base_name in ("jax_b1", f"jax_b{bucket}"):
            if base_name in got and base_name != name:
                row[f"vs_{base_name}"] = compare(r, got[base_name])
        out["arms"][name] = row
        OUT.write_text(json.dumps(out, indent=1, default=float))
        print(name, json.dumps({k: row[k] for k in ("loss", "loss_finite", "n_real",
                                                   "mean_plddt_real", "grads_nonfinite")}),
              flush=True)
        for base_name in ("jax_b1", f"jax_b{bucket}"):
            if f"vs_{base_name}" in row:
                print(f"   vs {base_name}:", json.dumps(row[f"vs_{base_name}"]), flush=True)
    print("bucket32_step complete", flush=True)


if __name__ == "__main__":
    main()

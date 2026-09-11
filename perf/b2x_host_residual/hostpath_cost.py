#!/usr/bin/env python3
"""Every host stage of a Boltz-2 fold that is NOT inside a `tt_bio.tenstorrent` module.

No device, no ttnn: that is the point. `b2x-op-cost-curve` bracketed the fold by
`tt_bio.tenstorrent` module class and found 3.212 s of a 23.710 s fold outside every one of
them, with 79 ttnn calls and zero device bytes. Those 3.212 s are the stages measured here, and
they all run in torch on the CPU, so they can be measured on any host with no card at all.

Runs the real code on the real fixture: `prepare_features` on `cdk2x2_512.yaml` + its 35-row
a3m, then the shipped `Boltz2` modules the trunk hands off to, at the shapes that fold produces.
`s`/`z` are synthetic (the trunk output would need a card) but their shapes and dtypes come from
the model config, and every stage timed here is shape-bound, not value-bound.

Stages:
  prepare              parse + MSA resolve + tokenize + featurize
  to_batch             unsqueeze/move
  input_embedder       host torch, once per fold
  rel_pos              RelativePositionEncoder, host torch, once per fold, O(n^2 * 139)
  diffusion_cond       DiffusionConditioning: pairwise_conditioner + atom_encoder + the three
                       bias stacks. NOT PORTED -- `tt_bio.tenstorrent` has no
                       DiffusionConditioning and no PairwiseConditioning.
  write_result         CIF write + metrics
The sampler loop's per-step host arithmetic is measured separately by sampler_host_cost.py.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = os.environ.get("B2X_REPO", "/home/moritz/tt-bio")
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(REPO) / "scripts" / "gpu_vs_tt"))

import torch  # noqa: E402

torch.set_grad_enabled(False)

OUT = {}
STAGES = defaultdict(float)
NCALL = defaultdict(int)


class Timer:
    """Nested wall-clock brackets, same contract as the on-card instrument."""

    def __init__(self):
        self.stack = []
        self.incl = defaultdict(float)
        self.child = defaultdict(float)
        self.n = defaultdict(int)
        self._undo = []

    def patch(self, owner, attr, label):
        orig = getattr(owner, attr)
        own = attr in getattr(owner, "__dict__", {})

        def w(*a, **k):
            return self.call(label, orig, a, k)
        setattr(owner, attr, w)
        self._undo.append((owner, attr, orig, own))

    def call(self, label, fn, a, k):
        path = "/".join(self.stack + [label])
        self.stack.append(label)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            self.stack.pop()
            self.incl[path] += dt
            self.n[path] += 1
            if self.stack:
                self.child["/".join(self.stack)] += dt

    def remove(self):
        for owner, attr, orig, own in reversed(self._undo):
            if own:
                setattr(owner, attr, orig)
            else:
                try:
                    delattr(owner, attr)
                except AttributeError:
                    setattr(owner, attr, orig)
        self._undo = []

    def table(self):
        return {p: {"calls": self.n[p], "incl_s": round(self.incl[p], 5),
                    "excl_s": round(self.incl[p] - self.child.get(p, 0.0), 5)}
                for p in sorted(self.incl)}


def timeit(fn, reps, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), [round(x, 5) for x in ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    import tt_baseline as B
    from tt_bio.main import (_resolve_recycling_steps, prepare_features, to_batch,
                             write_result)
    from tt_bio.worker import _ensure_local_artifacts
    from tt_bio.data.featurizer import Boltz2Featurizer
    from tt_bio.data.mol import load_canonicals
    from tt_bio.data.tokenize import Boltz2Tokenizer

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(Path(REPO) / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    fix = Path(REPO) / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    work = Path(os.environ.get("B2X_WORK", "/tmp/b2x_hostpath"))
    msa_dir, struct_dir = work / f"msa_{a.size}", work / "out"
    msa_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(model="boltz2", fast=False, output_format="cif",
               recycling_steps=B.RECYCLING_STEPS, sampling_steps=200, diffusion_samples=1,
               seed=0, trace=False, msa_dir=str(msa_dir), struct_dir=str(struct_dir),
               use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
               single_sequence=False, msa_server_url="https://api.colabfold.com",
               msa_pairing_strategy="greedy", msa_server_username=None,
               msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
               write_pae=False, write_pde=False, write_embeddings=False, method=None)
    _ensure_local_artifacts(cfg)
    n_msa = B.seed_msa_cache(tgt, a3m, msa_dir)

    OUT["env"] = {"host": os.uname().nodename, "torch": torch.__version__,
                  "threads": torch.get_num_threads(), "cpus": os.cpu_count(),
                  "size": a.size, "n_msa": n_msa, "reps": a.reps,
                  "repo": REPO, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "note": "CPU only, no device. s/z are synthetic at the model's shapes."}
    print(f"host {OUT['env']['host']}  torch {torch.__version__}  "
          f"threads {torch.get_num_threads()}  n_msa {n_msa}", flush=True)

    tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
    mol_dir = Path(cfg["mol_dir"])
    ccd = load_canonicals(mol_dir)

    def prep():
        return prepare_features(
            tgt, ccd=ccd, mol_dir=mol_dir, msa_dir=msa_dir, tokenizer=tokenizer,
            featurizer=featurizer, use_msa=True, msa_url=cfg["msa_server_url"],
            msa_strategy="greedy", msa_user=None, msa_pass=None, api_key=None,
            max_msa=8192, msa_db_path=None, use_envdb=False, single_sequence=False)

    t_prep, all_prep = timeit(lambda: prep(), a.reps)
    feats, input_struct = prep()
    n_tok = int(feats["token_pad_mask"].shape[0])
    n_atom = int(feats["atom_pad_mask"].shape[0])
    OUT["shapes"] = {"n_tokens": n_tok, "n_atoms": n_atom,
                     "msa_rows": (list(feats["msa"].shape) if "msa" in feats else None)}
    print(f"  prepare        {1e3*t_prep:9.2f} ms   n_tokens {n_tok}  n_atoms {n_atom}",
          flush=True)

    dev = torch.device("cpu")
    t_batch, all_batch = timeit(lambda: to_batch(feats, dev), a.reps)
    batch = to_batch(feats, dev)
    print(f"  to_batch       {1e3*t_batch:9.2f} ms", flush=True)

    # --- the shipped modules the trunk hands off to -----------------------------------------
    from tt_bio.boltz2 import Boltz2

    t0 = time.perf_counter()
    # `patch_boltz2_cfg` injects conf_kwargs inside `_WorkerState.load_model`; this harness
    # never calls it, so route the checkpoint load through a throwaway state to get exactly
    # the production hyperparameters rather than a second copy of them here.
    # use_tenstorrent=False: the three stages timed here (rel_pos, input_embedder,
    # diffusion_conditioning) are plain torch on either setting -- `tt_bio.tenstorrent` has no
    # DiffusionConditioning and no PairwiseConditioning at all -- and asking for the device
    # build would open a card this harness has no grant for.
    _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
                  "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
                  "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
                  "alignment_reverse_diff": True, "synchronize_sigmas": True}
    conf_kwargs = dict(
        predict_args={"recycling_steps": B.RECYCLING_STEPS, "sampling_steps": 200,
                      "diffusion_samples": 1, "max_parallel_samples": None},
        diffusion_process_args=_diffusion,
        pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
        msa_args={"subsample_msa": True, "num_subsampled_msa": 1024,
                  "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4,
                  "msa_dropout": 0.15, "z_dropout": 0.25, "pairwise_head_width": 32,
                  "pairwise_num_heads": 4, "activation_checkpointing": True},
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": True, "num_particles": 3,
                       "fk_lambda": 4.0, "fk_resampling_interval": 3, "num_gd_steps": 20},
        use_kernels=False, use_tenstorrent=False, trace=False, diffusion_trace=False)
    model = Boltz2.load_from_checkpoint(cfg["conf_ckpt"], **conf_kwargs).eval()
    OUT["model_load_s"] = round(time.perf_counter() - t0, 3)
    print(f"  (model load    {OUT['model_load_s']:9.2f} s, not part of a warm fold)", flush=True)

    token_s = model.diffusion_conditioning.atom_encoder.s_to_c_trans[0].normalized_shape[0]
    t_rel, all_rel = timeit(lambda: model.rel_pos(batch), a.reps)
    relpos = model.rel_pos(batch)
    token_z = relpos.shape[-1]
    print(f"  rel_pos        {1e3*t_rel:9.2f} ms   -> {tuple(relpos.shape)}", flush=True)

    t_inp, all_inp = timeit(lambda: model.input_embedder(batch), a.reps)
    s_inputs = model.input_embedder(batch)
    print(f"  input_embedder {1e3*t_inp:9.2f} ms   -> {tuple(s_inputs.shape)}", flush=True)

    # The trunk output is what `diffusion_conditioning` consumes; its widths come from the
    # modules that read it, not from a guess. Values are synthetic, shapes are the model's.
    z_trunk = torch.randn(1, n_tok, n_tok, token_z)
    s_tr = torch.randn(1, n_tok, token_s)
    OUT["probe"] = {"token_s": token_s, "token_z": token_z,
                    "s_inputs_w": int(s_inputs.shape[-1])}

    tm = Timer()
    import tt_bio.boltz2 as BZ
    tm.patch(BZ.PairwiseConditioning, "forward", "pairwise_conditioner")
    tm.patch(BZ.AtomEncoder, "forward", "atom_encoder")
    tm.patch(BZ.Transition, "forward", "transition")

    def dc():
        return model.diffusion_conditioning(
            s_trunk=s_tr, z_trunk=z_trunk, relative_position_encoding=relpos, feats=batch)

    t_dc, all_dc = timeit(lambda: dc(), a.reps)
    q, c, to_keys, aeb, adb, ttb = dc()
    sub = tm.table()
    tm.remove()
    print(f"  diffusion_cond {1e3*t_dc:9.2f} ms   token_trans_bias {tuple(ttb.shape)}",
          flush=True)
    for p, v in sorted(sub.items(), key=lambda kv: -kv[1]["excl_s"]):
        print(f"      {p:46s} incl {1e3*v['incl_s']/(a.reps+2):8.2f} ms  "
              f"excl {1e3*v['excl_s']/(a.reps+2):8.2f} ms  n {v['calls']}", flush=True)

    # --- write_result -----------------------------------------------------------------------
    mask = batch["atom_pad_mask"]
    coords = torch.randn(1, int(mask.sum().item()) if mask.dtype == torch.bool
                         else int(mask.shape[-1]), 3)
    coords = torch.randn(1, mask.shape[-1], 3)
    pred = {"exception": False, "masks": mask, "token_masks": batch["token_pad_mask"],
            "coords": coords, "plddt": torch.rand(1, mask.shape[-1]),
            "pde": torch.rand(1, n_tok, n_tok), "confidence_score": torch.rand(1),
            "complex_plddt": torch.rand(1), "complex_iplddt": torch.rand(1),
            "complex_pde": torch.rand(1), "complex_ipde": torch.rand(1)}
    t_wr, all_wr = timeit(lambda: write_result(pred, batch, input_struct, struct_dir, "cif",
                                               False, False, False), a.reps)
    print(f"  write_result   {1e3*t_wr:9.2f} ms", flush=True)

    OUT["stages_s"] = {
        "prepare": round(t_prep, 5), "to_batch": round(t_batch, 5),
        "rel_pos": round(t_rel, 5), "input_embedder": round(t_inp, 5),
        "diffusion_conditioning": round(t_dc, 5), "write_result": round(t_wr, 5),
    }
    OUT["stages_all_reps"] = {"prepare": all_prep, "to_batch": all_batch, "rel_pos": all_rel,
                              "input_embedder": all_inp, "diffusion_conditioning": all_dc,
                              "write_result": all_wr}
    OUT["diffusion_conditioning_subtree"] = sub
    OUT["sum_measured_s"] = round(sum(OUT["stages_s"].values()), 5)
    print(f"\n  SUM of the stages above: {OUT['sum_measured_s']:.3f} s "
          f"(the sampler loop is separate)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(OUT, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

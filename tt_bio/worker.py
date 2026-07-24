"""Long-lived prediction worker.

A worker process owns one accelerator slot for its entire lifetime: it loads the
Boltz-2 model once, then pulls jobs from a scheduler over HTTP and runs them
until cancelled. The same loop runs for local single-machine runs and for
multi-host runs; only the scheduler URL differs.
"""

from __future__ import annotations

import base64
import gc
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import torch

from tt_bio.distributed import ControllerClient, HttpProgressQueue


def _silence_subprocess_output() -> None:
    """Send stdout/stderr to /dev/null so kernel/library noise stays hidden."""
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    dn_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn_fd, 1)
    os.dup2(dn_fd, 2)
    os.close(dn_fd)


def _apply_tt_environment(worker_info: dict[str, Any]) -> None:
    """Configure TT visibility for this worker before importing ttnn."""
    if worker_info["accelerator"] != "tenstorrent":
        return
    os.environ["TT_VISIBLE_DEVICES"] = str(worker_info.get("visible_devices") or worker_info["device_id"])
    os.environ["TT_BIO_LOGICAL_DEVICE_ID"] = str(worker_info.get("logical_device_id", 0))
    mgd = worker_info.get("mesh_graph_descriptor")
    if mgd and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)


def _ensure_local_artifacts(cfg: dict[str, Any]) -> None:
    """Make sure model files and caches exist locally for this worker.

    Model checkpoints and the molecule library are always resolved to the
    worker's own ~/.boltz/ cache. For the MSA directory we prefer the path
    the controller asked for (so single-machine and shared-filesystem runs
    keep populating <out_dir>/msa/ exactly like the legacy pipeline) and
    only fall back to the local cache when that path is not writable on
    this host (the no-shared-FS multi-machine case).
    """
    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
    cache.mkdir(parents=True, exist_ok=True)
    # Protenix-v2: resolve the v2 checkpoint. Prefer $PROTENIX_CKPT, then the worker
    # cache, then download from the Hugging Face weights mirror on first use.
    if cfg.get("model") == "protenix-v2":
        from tt_bio.main import PROTENIX_REPO, download_mols, hf_artifact

        cfg["msa_dir"] = _resolve_msa_dir(cfg.get("msa_dir"), cache)
        cfg["protenix_ckpt"] = os.environ.get("PROTENIX_CKPT") or str(
            hf_artifact(PROTENIX_REPO, "protenix-v2.pt", cache))
        cfg["mol_dir"] = str(download_mols(cache))     # CCD templates for nucleic acids / ligands
        return
    # OpenDDE loads its weights from HF on the first fold.
    if cfg.get("model", "boltz2") in ("opendde", "opendde-abag"):
        cfg["opendde_ckpt"] = os.environ.get("OPENDDE_CKPT")
        return
    # ESMFold2 loads its weights from HF on the first fold and needs no Boltz-2
    # checkpoints / molecule library — only a writable MSA dir.
    if cfg.get("model", "boltz2") in ("esmfold2", "esmfold2-fast"):
        cfg["msa_dir"] = _resolve_msa_dir(cfg.get("msa_dir"), cache)
        return
    # ESMC embedding: weights come straight from the HF cache (load_esmc), no
    # Boltz-2 checkpoints/molecule library/MSA dir needed.
    if _is_esmc_model(cfg.get("model", "boltz2")):
        return
    from tt_bio.main import download_all

    download_all(cache)
    cfg["conf_ckpt"] = str(cache / "boltz2_conf.ckpt")
    cfg["aff_ckpt"] = str(cache / "boltz2_aff.ckpt")
    cfg["mol_dir"] = str(cache / "mols")
    cfg["msa_dir"] = _resolve_msa_dir(cfg.get("msa_dir"), cache)


def _resolve_msa_dir(requested: str | None, cache: Path) -> str:
    """Honor controller's msa_dir if it already exists and is writable on this
    host (covers single-machine runs and shared-filesystem multi-machine
    setups); otherwise fall back to ~/.boltz/msa/ on the worker."""
    if requested:
        path = Path(requested)
        if path.is_dir() and os.access(path, os.W_OK):
            return str(path)
    fallback = cache / "msa"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


def _is_esmc_model(model_id: str) -> bool:
    """True for any ESMC embedding model name (esmc-300m/600m/6b).

    Lazily imports tt_bio.esmc (which imports ttnn at module scope) so a
    worker that never handles embed jobs never pays that import cost.
    """
    from tt_bio.esmc import MODELS

    return model_id in MODELS


class _WorkerState:
    """Holds the loaded model and per-run helpers."""

    def __init__(self, accelerator: str) -> None:
        self.accelerator = accelerator
        self.run_id: str | None = None
        self.config_hash: str | None = None
        self.model_id: str | None = None   # the loaded model — reported to the
                                           # scheduler so it can keep this worker
                                           # on the same model (affinity).
        self.model = None
        self.aff_model = None
        self.prepare = None
        self.pfn = None  # progress callback (rebound per run)
        self._ccd = self._tokenizer = self._featurizer = self._mol_dir = None  # Boltz-2, cached
        if accelerator == "gpu" and torch.cuda.is_available():
            self.torch_device = torch.device("cuda:0")
        else:
            self.torch_device = torch.device("cpu")

    def configured_for(self, cfg: dict[str, Any]) -> bool:
        # Residency is keyed on the model setup only, NOT the run id: the loaded
        # weights are run-independent, so a resident model serves jobs from any
        # run/user of the same model with no reload. Per-run bits (output/MSA
        # paths, progress) are refreshed cheaply in bind_run().
        return self.model is not None and self.config_hash == _hash_run_config(cfg)

    def reset(self) -> None:
        self.model = None
        self.aff_model = None
        self.prepare = None
        self.pfn = None
        self.run_id = None
        self.config_hash = None
        self.model_id = None
        self._ccd = self._tokenizer = self._featurizer = self._mol_dir = None
        gc.collect()
        if self.accelerator == "tenstorrent":
            try:
                from tt_bio.tenstorrent import cleanup as _tt_cleanup

                _tt_cleanup()
            except Exception:
                pass

    def free_model(self) -> None:
        """Free the resident predict model but KEEP the device open.

        Used before running an in-process design shard: the shard reuses this
        worker's already-open chip and loads its own models, so we must drop the
        predict weights (free memory) WITHOUT closing the device. Closing and
        re-opening a chip per shard is exactly what deadlocked the UMD
        device-init path (see tenstorrent._device_init_lock); reusing one
        persistent open avoids it entirely."""
        self.model = None
        self.aff_model = None
        self.prepare = None
        self.pfn = None
        self.run_id = None
        self.config_hash = None
        self.model_id = None
        self._ccd = self._tokenizer = self._featurizer = self._mol_dir = None
        gc.collect()

    def load_model(self, cfg: dict[str, Any]) -> None:
        """Load the heavy model weights onto the device. Keyed on the model
        config (see configured_for), so it runs once per model, not once per run."""
        if self.accelerator == "tenstorrent":
            from tt_bio.tenstorrent import set_fast_mode

            set_fast_mode(cfg.get("fast", False))

        model_id = cfg.get("model", "boltz2")
        if model_id in ("esmfold2", "esmfold2-fast"):
            from tt_bio.esmfold2_runtime import load_ttnn_esmfold2

            repo = "biohub/ESMFold2-Fast" if model_id == "esmfold2-fast" else "biohub/ESMFold2"
            self.model = load_ttnn_esmfold2(esmfold2_repo=repo, fast=cfg.get("fast", False))
            self.model._esmc.preload()
        elif model_id == "protenix-v2":
            from tt_bio.protenix import Protenix

            self.model = Protenix.load_from_checkpoint(cfg["protenix_ckpt"])
        elif model_id in ("opendde", "opendde-abag"):
            from tt_bio.opendde import OpenDDE

            self.model = OpenDDE.load_from_checkpoint(
                cfg.get("opendde_ckpt"), abag=(model_id == "opendde-abag"))
        elif _is_esmc_model(model_id):
            from tt_bio.esmc import load_esmc

            self.model = load_esmc(model_id, fast=cfg.get("fast", False))
        else:
            from tt_bio.boltz2 import Boltz2
            from tt_bio.data.featurizer import Boltz2Featurizer
            from tt_bio.data.mol import load_canonicals
            from tt_bio.data.tokenize import Boltz2Tokenizer

            self._tokenizer, self._featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
            self._mol_dir = Path(cfg["mol_dir"])
            self._ccd = load_canonicals(self._mol_dir)
            self.model = (
                Boltz2.load_from_checkpoint(cfg["conf_ckpt"], **cfg["conf_kwargs"])
                .eval()
                .to(self.torch_device)
            )
        self.config_hash = _hash_run_config(cfg)
        self.model_id = model_id

    def bind_run(self, run_id: str, cfg: dict[str, Any]) -> None:
        """Cheap per-run rebinding so a resident model serves a new run/user
        correctly: point Boltz-2's featurizer at this run's MSA/output dirs.
        ESMFold2 / Protenix read those straight from cfg in predict_one."""
        self.run_id = run_id
        if self.model_id == "boltz2":
            from tt_bio.main import prepare_features

            self.prepare = partial(
                prepare_features,
                ccd=self._ccd, mol_dir=self._mol_dir, msa_dir=Path(cfg["msa_dir"]),
                tokenizer=self._tokenizer, featurizer=self._featurizer,
                use_msa=cfg["use_msa_server"], msa_url=cfg["msa_server_url"],
                msa_strategy=cfg["msa_pairing_strategy"], msa_user=cfg["msa_server_username"],
                msa_pass=cfg["msa_server_password"], api_key=cfg["api_key_value"],
                max_msa=cfg["max_msa_seqs"], msa_db_path=cfg.get("msa_db_path"),
                use_envdb=cfg.get("use_envdb", False),
                single_sequence=cfg.get("single_sequence", False),
            )
        else:
            self.prepare = None

    def _maybe_ref_bf16(self):
        """Integration-parity envelope (scripts/full_parity_gate.py): when TT_BIO_REF_BF16=1 and
        this is the CPU/host reference (NOT tenstorrent), run the model forward under a bf16
        autocast so its closed-loop divergence from the fp32 reference measures the intrinsic
        bf16 cost of the full sampler trajectory (chaotic amplification included). Applied at
        every forward the device runs in bf16 — the structure ``predict_step`` AND the affinity
        ``aff_model.predict_step`` (the device runs the affinity head in bf16 too, unless
        BOLTZ2_AFFINITY_DIFFUSION_FP32_DEVICE=1) — so the bf16 reference mirrors the device's
        dtype boundary rather than leaving the affinity scalar in fp32. Shared draws are
        preserved: the diffusion ``torch.randn`` draws (boltz2.py:4092/4127) run on CPU MT19937
        from the one seed, unaffected by autocast, so fp32 and bf16 references differ only in
        arithmetic dtype, nothing stochastic. Default off — device runs and the fp32 reference
        get a nullcontext and are untouched."""
        import contextlib
        _on = os.environ.get("TT_BIO_REF_BF16", "0") not in ("0", "")
        if _on and self.accelerator != "tenstorrent":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def predict_one(self, path: Path, cfg: dict[str, Any]):
        if cfg.get("model") in ("opendde", "opendde-abag"):
            return self._predict_opendde_one(path, cfg)
        if cfg.get("model") == "protenix-v2":
            return self._predict_protenix_one(path, cfg)
        if cfg.get("model", "boltz2") in ("esmfold2", "esmfold2-fast"):
            return self._predict_esmfold2_one(path, cfg)
        if _is_esmc_model(cfg.get("model", "boltz2")):
            return self._predict_embed_one(path, cfg)

        from tt_bio.main import to_batch, write_result

        # The boltz-2 path calls ``predict_step`` directly (unlike the esmfold2 /
        # protenix / opendde paths, which re-seed via ``_seed_context`` inside
        # ``fold_complex``). This worker is spawned with ``mp.get_context(
        # "spawn")``, so the controller's ``torch.manual_seed(seed)`` does NOT
        # propagate here, and the boltz-2 forward never re-seeds on its own. The
        # official ``boltz`` reference calls ``seed_everything(seed)`` once at
        # the start of ``predict`` and then runs structure -> affinity from that
        # one global RNG stream, so the affinity diffusion's ``torch.randn``
        # draws are reproducible. Without this seed the device's affinity value
        # swings ~0.05 log10(IC50) between identical-seed runs (verified: two
        # seed-0 runs gave -0.394 vs -0.440, a 0.047 spread larger than the whole
        # FKBP12 GAP of 0.041 and the reference floor R=0.010), which the tight
        # affinity floor catches as a GAP. Seed once here (before the structure
        # forward) and do NOT re-seed before ``predict_affinity`` so the device
        # matches the reference's single-seed structure->affinity RNG stream.
        _seed = cfg.get("seed")
        if _seed is not None:
            import random as _random
            import numpy as _np
            _random.seed(_seed)
            _np.random.seed(_seed)
            torch.manual_seed(_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_seed)

        feats, input_struct = self.prepare(path, method=cfg.get("method"), progress=self.pfn)
        batch = to_batch(feats, self.torch_device)
        with torch.no_grad():
            with self._maybe_ref_bf16():
                pred = self.model.predict_step(batch)
        metrics, best = write_result(
            pred,
            batch,
            input_struct,
            Path(cfg["struct_dir"]),
            cfg["output_format"],
            cfg["write_pae"],
            cfg["write_pde"],
            cfg["write_embeddings"],
        )
        return metrics, best, feats

    def _predict_esmfold2_one(self, path: Path, cfg: dict[str, Any]):
        import hashlib
        import types

        from tt_bio.esmfold2 import report_progress
        from tt_bio.esmfold2_runtime import fold_complex, resolve_msa
        from tt_bio.main import _generate_esmfold2_a3m, _read_protein_chains, _write_structure

        chains = _read_protein_chains(path)
        if not chains:
            raise RuntimeError("no protein sequences")
        msa_dir = Path(cfg["msa_dir"])
        max_msa = cfg.get("max_msa_seqs") or 16384
        # Only the checkpoints that ship an MSA encoder can use an MSA. ESMFold2
        # has one; ESMFold2-Fast does not (model.msa_encoder is None), so there's
        # nothing to consume an alignment — skip the search and fold single-seq
        # rather than do wasted work and falsely report msa=true.
        uses_msa = getattr(self.model, "msa_encoder", None) is not None

        # MSA phase — rendered as the "MSA" stage, exactly like Boltz-2 (which
        # generates worker-side in prepare_features). When a source is given we
        # search any chain whose {seq_hash}.a3m/.csv is not already cached, into
        # the shared msa_dir. MSA is optional: with no source, fold single-seq.
        report_progress("msa")
        if uses_msa and (cfg.get("use_msa_server") or cfg.get("msa_db_path") or cfg.get("msa_endpoint")):
            to_gen = {}
            for _cid, seq, spec in chains:
                if spec and Path(spec).expanduser().exists():
                    continue
                h = hashlib.sha256(seq.encode()).hexdigest()[:16]
                if not (msa_dir / f"{h}.a3m").exists() and not (msa_dir / f"{h}.csv").exists():
                    to_gen[h] = seq
            if to_gen:
                _generate_esmfold2_a3m(
                    to_gen, path.stem, msa_dir, cfg.get("msa_db_path"), cfg.get("use_envdb", False),
                    cfg.get("msa_server_url"), cfg.get("msa_pairing_strategy"),
                    cfg.get("msa_server_username"), cfg.get("msa_server_password"),
                    cfg.get("api_key_value"), msa_endpoint=cfg.get("msa_endpoint"))

        report_progress("prep")
        chains = [(cid, seq, resolve_msa(spec, seq, msa_dir, max_sequences=max_msa) if uses_msa else None)
                  for cid, seq, spec in chains]
        res = fold_complex(
            self.model, chains,
            num_loops=cfg["recycling_steps"], num_sampling_steps=cfg["sampling_steps"],
            num_diffusion_samples=cfg["diffusion_samples"], seed=cfg.get("seed") or 0,
        )
        out = Path(cfg["struct_dir"]) / f"{path.stem}.{cfg['output_format']}"
        _write_structure(res.complex, out, cfg["output_format"])
        metrics = {
            "plddt": round(float(res.plddt.mean()), 4),
            "n_residues": sum(len(c[1]) for c in chains), "n_chains": len(chains),
            "msa": any(c[2] is not None for c in chains),
            "samples": cfg["diffusion_samples"],  # best-of-N: report N (plddt is the winner's)
        }
        if getattr(res, "ptm", None) is not None:
            metrics["ptm"] = round(float(res.ptm), 4)
        # _execute_job inspects feats["record"].affinity; ESMFold2 has no affinity.
        feats = {"record": types.SimpleNamespace(affinity=False)}
        return metrics, None, feats

    def _predict_opendde_one(self, path: Path, cfg: dict[str, Any]):
        """OpenDDE protein co-fold: sequence(s) -> (optional per-chain MSA) -> on-device
        structural-token fold -> structure. Rides the SAME MSA stage as Protenix-v2 /
        ESMFold2 / Boltz-2: each protein chain whose {seq_hash}.a3m is not cached is
        searched into the shared msa_dir, resolved, and featurized via
        build_complex_features' block-diagonal MSA. Protein + ligand co-folds (nucleic-acid
        structural tokens not ported yet). Ligand atoms are tokenized per-atom by
        build_complex_features and expand to one "atom"-role structural token each
        (opendde_data.build_structural_token_features), so a covalent inhibitor bonded
        to a protein Cys is honored end-to-end. Confidence-based best-of-N ranking and
        CIF writing reuse Protenix-v2's machinery verbatim (OpenDDE.fold rides the same
        ConfidenceHead / build_complex_features / _write_protenix_structure)."""
        import hashlib
        import types

        from tt_bio.esmfold2 import report_progress
        from tt_bio.main import (_generate_esmfold2_a3m,
                                 _generate_opendde_paired_a3m, _read_bio_chains,
                                 _read_bio_constraints, _resolve_a3m_text,
                                 _write_protenix_structure)
        from tt_bio.protenix_data import build_complex_features

        chains = _read_bio_chains(path)
        if not chains:
            raise RuntimeError("no protein sequences")
        unsupported = [cid for cid, _s, _sp, mt in chains if mt not in ("protein", "ligand")]
        if unsupported:
            raise RuntimeError(
                f"--model opendde supports protein + ligand chains only (chain(s) "
                f"{unsupported} are nucleic-acid); nucleic-acid structural tokens are not "
                "ported yet. Ligand covalent bonds are honored.")
        bonds = _read_bio_constraints(path)
        msa_dir = Path(cfg["msa_dir"])

        report_progress("msa")
        # search any uncached protein chain (batched into one MSA call), reusing the
        # Protenix-v2 / ESMFold2 stage verbatim -- no separate OpenDDE MSA path.
        # A second, paired (species-pairing) search is run below for multi-chain
        # complexes to inject the cross-chain co-evolution signal.
        want_msa = cfg.get("use_msa_server") or cfg.get("msa_db_path") or cfg.get("msa_endpoint")
        need = {}
        for _cid, cseq, spec, mt in chains:
            have_spec = bool(spec and Path(spec).expanduser().exists())
            if mt == "protein" and want_msa and not have_spec:
                h = hashlib.sha256(cseq.encode()).hexdigest()[:16]
                if not (msa_dir / f"{h}.a3m").exists():
                    need[h] = cseq
        if need:
            _generate_esmfold2_a3m(
                need, path.stem, msa_dir, cfg.get("msa_db_path"),
                cfg.get("use_envdb", False), cfg.get("msa_server_url"),
                cfg.get("msa_pairing_strategy"), cfg.get("msa_server_username"),
                cfg.get("msa_server_password"), cfg.get("api_key_value"),
                msa_endpoint=cfg.get("msa_endpoint"))
        chain_specs = [(cseq, _resolve_a3m_text(spec, cseq, msa_dir), mt)
                       for _cid, cseq, spec, mt in chains]

        # Paired (species-pairing) MSA for multi-chain complexes -- the cross-chain
        # co-evolution signal the reference OpenDDE pipeline injects via
        # MSAPairingEngine.pair_chains_by_species and this port otherwise lacks
        # (unpaired block-diagonal MSA carries no cross-chain signal). Best-effort:
        # a failed paired search falls back to unpaired-only so the fold still runs.
        paired_a3ms = None
        n_prot = sum(1 for _c, _s, _sp, mt in chains if mt == "protein")
        if n_prot > 1 and want_msa:
            paired_seqs = {hashlib.sha256(cseq.encode()).hexdigest()[:16]: cseq
                           for _cid, cseq, _spec, mt in chains if mt == "protein"}
            try:
                paired = _generate_opendde_paired_a3m(
                    paired_seqs, path.stem, msa_dir, cfg.get("msa_server_url"),
                    cfg.get("msa_pairing_strategy"), cfg.get("msa_server_username"),
                    cfg.get("msa_server_password"), cfg.get("api_key_value"),
                    msa_db_path=cfg.get("msa_db_path"), use_envdb=cfg.get("use_envdb", False))
                paired_a3ms = [paired.get(hashlib.sha256(cseq.encode()).hexdigest()[:16])
                               for _cid, cseq, _spec, mt in chains if mt == "protein"]
            except Exception as e:  # noqa: BLE001 -- best-effort, fall back to unpaired
                print(f"paired MSA search failed ({e!r}); folding unpaired-only", file=sys.stderr)
                paired_a3ms = None

        report_progress("prep")
        feats = build_complex_features(chain_specs, chain_ids=[cid for cid, _s, _sp, _mt in chains],
                                       bonds=bonds, paired_a3ms=paired_a3ms)

        # OpenDDE.fold rides the Protenix-v2 trunk + EDM sampler, so the same
        # progress_fn path reports trunk iterations and diffusion steps — no
        # separate OpenDDE progress wiring, and no premature "diffusion" emit
        # that would skip the trunk phase on the live view.
        n_sample = int(cfg["diffusion_samples"])
        # Integration-parity envelope: run the bf16 CPU reference fold under bf16
        # autocast (see _predict_protenix_one / _maybe_ref_bf16). nullcontext on
        # device and on the fp32 reference, so those paths are untouched.
        with torch.no_grad(), self._maybe_ref_bf16():
            coords, conf = self.model.fold(
                feats, n_step=cfg["sampling_steps"], n_sample=n_sample,
                seed=cfg.get("seed") or 0, progress_fn=report_progress,
                n_cycles=cfg.get("recycling_steps"), trace=cfg.get("trace", False),
                return_confidence=True)
        confs = conf if isinstance(conf, list) else [conf]

        # AF-style ranking score: ipTM-weighted for complexes, pTM for monomers, falling
        # back to pLDDT only if neither is available -- identical to Protenix-v2's ranking.
        def _score(c):
            ptm, iptm = c.get("ptm", 0.0), c.get("iptm", 0.0)
            if iptm > 0.0:
                return 0.8 * iptm + 0.2 * ptm
            return ptm if ptm > 0.0 else c["plddt"]

        order = sorted(range(len(confs)), key=lambda k: _score(confs[k]), reverse=True)
        rank_of = {k: r for r, k in enumerate(order)}

        struct_dir = Path(cfg["struct_dir"])
        stem, fmt = path.stem, cfg["output_format"]
        for k in range(len(confs)):
            r = rank_of[k]
            name = f"{stem}.{fmt}" if r == 0 else f"{stem}_model_{r}.{fmt}"
            _write_protenix_structure(coords[k], feats, None, struct_dir / name, fmt,
                                      b_factors=confs[k]["plddt_atom"] * 100.0)

        def _row(c):
            return {"complex_plddt": round(c["plddt"], 6), "plddt": round(c["plddt"], 6),
                    "ptm": round(c.get("ptm", 0.0), 6), "iptm": round(c.get("iptm", 0.0), 6),
                    "confidence_score": round(_score(c), 6)}

        best = confs[order[0]]
        metrics = {
            **_row(best),
            "n_residues": sum(len(cseq) for _c, cseq, _s, mt in chains if mt != "ligand"),
            "n_chains": len(chains), "n_tokens": int(feats["restype"].shape[0]),
            "msa": any(a for _, a, _ in chain_specs), "n_atoms": int(coords.shape[1]),
            "samples": n_sample,
        }
        if len(confs) > 1:
            metrics["all_runs"] = [{"rank": rank_of[k], **_row(confs[k])} for k in order]
        if cfg.get("write_pae"):                       # token-token PAE/PDE of the best sample
            import numpy as np
            np.savez(struct_dir / f"{stem}_pae.npz",
                     pae=best["pae"].numpy(), pde=best["pde"].numpy())
        return metrics, None, {"record": types.SimpleNamespace(affinity=False)}

    def _predict_protenix_one(self, path: Path, cfg: dict[str, Any]):
        """Protenix-v2 protein fold: sequence(s) -> (optional per-chain MSA) -> on-device fold
        -> structure. Rides the same MSA stage as ESMFold2/Boltz-2: each chain whose
        {seq_hash}.a3m is not cached is searched into the shared msa_dir, resolved, and
        featurized. Multi-chain inputs fold as a true complex (per-chain asym/entity/sym +
        block-diagonal MSA via build_complex_features)."""
        import hashlib
        import types

        from tt_bio.esmfold2 import report_progress
        from tt_bio.main import (_generate_esmfold2_a3m, _read_bio_chains,
                                 _read_bio_constraints, _resolve_a3m_text,
                                 _write_protenix_structure)
        from tt_bio.protenix_data import build_complex_features

        chains = _read_bio_chains(path)
        if not chains:
            raise RuntimeError("no protein/nucleic-acid sequences")
        bonds = _read_bio_constraints(path)   # covalent bonds; rejects pocket/contact
        msa_dir = Path(cfg["msa_dir"])

        report_progress("msa")
        # search any uncached protein chain (batched into one MSA call); NA chains are single-seq
        want_msa = cfg.get("use_msa_server") or cfg.get("msa_db_path") or cfg.get("msa_endpoint")
        need = {}
        for _cid, cseq, spec, mt in chains:
            have_spec = bool(spec and Path(spec).expanduser().exists())
            if mt == "protein" and want_msa and not have_spec:
                h = hashlib.sha256(cseq.encode()).hexdigest()[:16]
                if not (msa_dir / f"{h}.a3m").exists():
                    need[h] = cseq
        if need:
            _generate_esmfold2_a3m(
                need, path.stem, msa_dir, cfg.get("msa_db_path"),
                cfg.get("use_envdb", False), cfg.get("msa_server_url"),
                cfg.get("msa_pairing_strategy"), cfg.get("msa_server_username"),
                cfg.get("msa_server_password"), cfg.get("api_key_value"),
                msa_endpoint=cfg.get("msa_endpoint"))
        chain_specs = [(cseq, _resolve_a3m_text(spec, cseq, msa_dir) if mt == "protein" else None, mt)
                       for _cid, cseq, spec, mt in chains]

        report_progress("prep")
        feats = build_complex_features(chain_specs, mol_dir=cfg.get("mol_dir"),
                                       chain_ids=[cid for cid, _s, _sp, _mt in chains], bonds=bonds)

        # One shared progress path: report_progress has exactly the progress_fn
        # signature, so hand it straight to the model — trunk iterations report
        # as "trunk", diffusion steps as "diffusion" (no remapping that would
        # hide the trunk phase).
        n_sample = int(cfg["diffusion_samples"])
        # Integration-parity envelope: the bf16 CPU reference must run the whole
        # protenix fold under bf16 autocast (mirroring the boltz2 path at
        # predict_step), otherwise the bf16 ref runs in fp32, the envelope
        # denominator collapses to ~0 and any device residual reads as a false GAP.
        # On device (accelerator == "tenstorrent") and on the fp32 reference this
        # is a nullcontext, so those paths are untouched.
        with torch.no_grad(), self._maybe_ref_bf16():
            coords, conf = self.model.fold(
                feats, n_step=cfg["sampling_steps"], n_sample=n_sample,
                seed=cfg.get("seed") or 0, progress_fn=report_progress,
                return_confidence=True, n_cycles=cfg.get("recycling_steps"),
            )
        confs = conf if isinstance(conf, list) else [conf]

        # AF-style ranking score: ipTM-weighted for complexes, pTM for monomers,
        # falling back to pLDDT only if neither is available. Picks the best sample
        # and orders all_runs — mirrors Boltz-2's confidence_score ranking.
        def _score(c):
            ptm, iptm = c.get("ptm", 0.0), c.get("iptm", 0.0)
            if iptm > 0.0:
                return 0.8 * iptm + 0.2 * ptm
            return ptm if ptm > 0.0 else c["plddt"]

        order = sorted(range(len(confs)), key=lambda k: _score(confs[k]), reverse=True)
        rank_of = {k: r for r, k in enumerate(order)}    # sample index -> rank (0 = best)

        struct_dir = Path(cfg["struct_dir"])
        stem, fmt = path.stem, cfg["output_format"]
        # Write best as "{stem}.{fmt}" and the rest as "{stem}_model_{rank}.{fmt}",
        # exactly like Boltz-2's write_result, so the web portal's progress count,
        # ensemble-similarity and downloads treat both models identically.
        for k in range(len(confs)):
            r = rank_of[k]
            name = f"{stem}.{fmt}" if r == 0 else f"{stem}_model_{r}.{fmt}"
            # per-atom pLDDT (0-1) -> B-factors (0-100), the AF/Boltz convention
            _write_protenix_structure(coords[k], feats, None, struct_dir / name, fmt,
                                      b_factors=confs[k]["plddt_atom"] * 100.0)

        def _row(c):
            return {"complex_plddt": round(c["plddt"], 6), "plddt": round(c["plddt"], 6),
                    "ptm": round(c.get("ptm", 0.0), 6), "iptm": round(c.get("iptm", 0.0), 6),
                    "confidence_score": round(_score(c), 6)}

        best = confs[order[0]]
        metrics = {
            **_row(best),
            "n_residues": sum(len(cseq) for _c, cseq, _s, mt in chains if mt != "ligand"),
            "n_chains": len(chains), "n_tokens": int(feats["restype"].shape[0]),
            "msa": any(a for _, a, _ in chain_specs),
            "n_atoms": int(coords.shape[1]), "samples": n_sample,
        }
        if len(confs) > 1:
            metrics["all_runs"] = [{"rank": rank_of[k], **_row(confs[k])} for k in order]
        if cfg.get("write_pae"):                       # token-token PAE/PDE of the best sample
            import numpy as np
            np.savez(struct_dir / f"{stem}_pae.npz",
                     pae=best["pae"].numpy(), pde=best["pde"].numpy())
        return metrics, None, {"record": types.SimpleNamespace(affinity=False)}

    def _predict_embed_one(self, path: Path, cfg: dict[str, Any]):
        """Embed one job's shard of sequences with the resident ESMC model.

        ``path`` is a YAML ``{id: sequence}`` mapping (one shard of a larger
        --controller embed run). Writes per-sequence ``.npz`` (or one shard
        parquet, named by job id to avoid colliding with other shards' output
        once every job's outputs land in the same directory) into struct_dir —
        the same output-shipping path predict/design jobs already use.
        """
        import types

        from tt_bio.esmc import embed_sequences, load_sequences, write_npz, write_parquet

        sequences = load_sequences(path)
        results = embed_sequences(
            self.model, sequences, return_logits=cfg.get("return_logits", False),
            pool=cfg.get("pool", "mean"), batch_size=cfg.get("batch_size", 8),
        )
        struct_dir = Path(cfg["struct_dir"])
        if cfg.get("output_format") == "parquet":
            write_parquet(results, struct_dir / f"{cfg['job_id']}.parquet")
        else:
            for emb in results:
                write_npz(emb, struct_dir / f"{emb.id}.npz")
        metrics = {
            "n_sequences": len(results),
            "d_model": int(results[0].pooled.shape[0]) if results else 0,
            "ids": [e.id for e in results],
            "lengths": [len(e.sequence) for e in results],
        }
        return metrics, None, {"record": types.SimpleNamespace(affinity=False)}

    def predict_affinity(self, path: Path, pred_structure, cfg: dict[str, Any]) -> dict[str, float]:
        from tt_bio.boltz2 import Boltz2
        from tt_bio.main import to_batch

        if self.aff_model is None:
            from tt_bio.tenstorrent import affinity_diffusion_fp32_device

            fp32_device = (
                os.environ.get("BOLTZ2_AFFINITY_DIFFUSION_FP32_DEVICE", "0") == "1"
            )
            with affinity_diffusion_fp32_device(fp32_device):
                self.aff_model = (
                    Boltz2.load_from_checkpoint(cfg["aff_ckpt"], **cfg["aff_kwargs"])
                    .eval()
                    .to(self.torch_device)
                )

        feats, _ = self.prepare(path, method="other", affinity=True, pred_structure=pred_structure)
        batch = to_batch(feats, self.torch_device)
        with torch.no_grad():
            with self._maybe_ref_bf16():
                pred = self.aff_model.predict_step(batch)
        if pred.get("exception"):
            return {}
        keys = [
            "affinity_pred_value",
            "affinity_probability_binary",
            "affinity_pred_value1",
            "affinity_probability_binary1",
            "affinity_pred_value2",
            "affinity_probability_binary2",
        ]
        return {k: round(pred[k].item(), 6) for k in keys if k in pred}


def _hash_run_config(cfg: dict[str, Any]) -> str:
    """Stable hash of the parts of the config that affect model setup."""
    import hashlib
    import json

    keep = {k: cfg.get(k) for k in ("model", "conf_kwargs", "aff_kwargs", "fast", "method")}
    blob = json.dumps(keep, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _install_signal_handlers() -> None:
    def _raise(signum, _frame):
        raise KeyboardInterrupt(f"worker received signal {signum}")

    try:
        signal.signal(signal.SIGTERM, _raise)
        signal.signal(signal.SIGINT, _raise)
    except Exception:
        pass


def run_worker_loop(
    controller_url: str,
    worker_info: dict[str, Any],
    debug: bool = False,
    idle_poll: float = 1.0,
) -> None:
    """Connect to a scheduler and process jobs until cancelled.

    Loads model artifacts once per run and reuses them for every job in that
    run. If the run's config changes, the model is reloaded.
    """
    if not debug:
        _silence_subprocess_output()
    _install_signal_handlers()
    _apply_tt_environment(worker_info)

    client = ControllerClient(controller_url)
    worker_id = worker_info["worker_id"]
    meta = {
        "dev": worker_info["device_id"],
        "worker": worker_id,
        "host": worker_info["host"],
        "accelerator": worker_info["accelerator"],
        "label": worker_info["label"],
    }

    # Background heartbeat: while the main loop is blocked computing (MSA fetch,
    # folding, a design-shard subprocess) it isn't leasing, so without this the
    # controller would mark a perfectly healthy worker offline. A daemon thread
    # pings the controller so a worker counts as online whenever its process is.
    import threading
    _stop_beat = threading.Event()

    def _heartbeat_loop():
        while not _stop_beat.wait(8.0):
            try:
                client.heartbeat(worker_info)
            except Exception:
                pass

    threading.Thread(target=_heartbeat_loop, daemon=True).start()

    def emit(run_id: str, event: str, **kw):
        try:
            client.event(run_id, worker_id, {"event": event, **meta, **kw})
        except Exception:
            pass

    state = _WorkerState(worker_info["accelerator"])
    # Open this worker's chip once, now, while the fleet is quiescent (startup),
    # and keep it open for every job — predict AND design. Every device open then
    # happens at startup, never during active operation, which is what keeps us
    # off the UMD concurrent-device-init deadlock (see tenstorrent._device_init_lock).
    if state.accelerator == "tenstorrent":
        try:
            from tt_bio.tenstorrent import get_device as _get_device
            _get_device()
        except Exception:
            traceback.print_exc()
            # The chip didn't come up with working local dispatch (e.g. a raced
            # "remote-only" bring-up). Do NOT stay online serving jobs we'd fail:
            # exit so the pool supervisor respawns us. The respawn reopens under the
            # host-wide device-init lock (one chip at a time), which is exactly what
            # clears the concurrent-init race behind a bad bring-up.
            return
    try:
        while True:
            # Tolerate a controller that's briefly unreachable (restart, network
            # blip): retry leasing instead of crashing the worker. This makes the
            # fleet self-healing — a worker reconnects on its own when the
            # controller comes back, with no manual restart.
            try:
                # Tell the scheduler which model we already have resident so it
                # can keep us on it (affinity) and avoid a reload.
                worker_info["model"] = state.model_id
                lease = client.lease(worker_info, batch_size=1)
            except Exception:
                time.sleep(idle_poll)
                continue
            jobs = lease.get("jobs") or []
            if not jobs:
                time.sleep(idle_poll)
                continue

            run_id = lease["run_id"]
            cfg = dict(lease["config"])

            # Design shards ride the same scheduler as prediction. They run the
            # BoltzGen single-device pipeline IN-PROCESS on this worker's already-
            # open chip — reusing the persistent device instead of cold-opening a
            # fresh one per shard (which raced the UMD device-init path and
            # deadlocked). Free the predict model first (memory) but keep the chip.
            if cfg.get("kind") == "design":
                state.free_model()
                for job in jobs:
                    _execute_design_job_inprocess(client, run_id, worker_id, worker_info, meta, job, cfg)
                continue

            _ensure_local_artifacts(cfg)

            try:
                # Reload weights only when the model actually changes — a resident
                # model is reused across runs/users of the same model.
                if not state.configured_for(cfg):
                    state.reset()
                    emit(run_id, "loading")
                    state.load_model(cfg)
                # Per-run rebinding (cheap) every job: output/MSA paths + a fresh
                # progress callback aimed at this run.
                state.bind_run(run_id, cfg)
                from tt_bio.progress import make_progress_fn

                pfn = make_progress_fn(
                    HttpProgressQueue(client, run_id, worker_id),
                    worker_info["device_id"], worker_id, meta,
                )
                state.pfn = pfn
                if cfg.get("model", "boltz2") == "boltz2":
                    state.model.progress_fn = pfn
                else:
                    from tt_bio import esmfold2 as _E

                    _E.set_progress(pfn)  # esmfold2 + protenix report via this module
            except Exception as exc:
                traceback.print_exc()
                _complete_failure(client, run_id, worker_id, meta, jobs, str(exc)[:200])
                state.reset()
                continue

            for job in jobs:
                _execute_job(state, job, cfg, run_id, client, worker_id, meta)
    except KeyboardInterrupt:
        pass
    finally:
        state.reset()


def _execute_job(
    state: _WorkerState,
    job: dict[str, Any],
    cfg: dict[str, Any],
    run_id: str,
    client: ControllerClient,
    worker_id: str,
    meta: dict[str, Any],
) -> None:
    job_id = job["id"]
    filename = job.get("name") or f"{job_id}.yaml"
    row: dict[str, Any] = {"id": job_id, "status": "failed"}
    t0 = time.time()

    def emit(event: str, **kw):
        try:
            client.event(run_id, worker_id, {"event": event, **meta, **kw})
        except Exception:
            pass

    workdir = Path(tempfile.mkdtemp(prefix=f"tt-bio-{job_id}-"))
    input_path = workdir / filename
    output_dir = workdir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    job_cfg = dict(cfg)
    job_cfg["struct_dir"] = str(output_dir)
    job_cfg["job_id"] = job_id

    outputs: dict[str, str] = {}
    emit("start", name=job_id)
    try:
        try:
            input_path.write_bytes(base64.b64decode(job.get("input_b64", "")))
        except Exception as exc:
            raise RuntimeError(f"failed to decode input bytes: {exc}") from exc

        # Both model families start in the MSA stage and resolve/search MSAs
        # worker-side; the esmfold2 path then reports "prep" before folding.
        emit("stage", stage="msa")
        metrics, best, feats = state.predict_one(input_path, job_cfg)
        emit("stage", stage="saving")
        if metrics:
            row.update(metrics)
            row["status"] = "ok"
            row["runtime_s"] = round(time.time() - t0, 1)
            if feats["record"].affinity and best is not None:
                try:
                    aff = state.predict_affinity(input_path, best, job_cfg)
                    row.update(aff)
                except Exception:
                    traceback.print_exc()
        outputs = _read_outputs(output_dir)
    except Exception as exc:
        traceback.print_exc()
        row["error"] = str(exc)[:200]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    try:
        client.complete(
            run_id,
            worker_id,
            row,
            {
                **meta,
                "event": "done",
                "name": job_id,
                "time": round(time.time() - t0, 1),
                "status": row["status"],
                "error": row.get("error", ""),
                "row": row,
            },
            outputs=outputs or None,
        )
    except Exception:
        traceback.print_exc()


def _execute_design_job_inprocess(
    client: ControllerClient,
    run_id: str,
    worker_id: str,
    worker_info: dict[str, Any],
    meta: dict[str, Any],
    job: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    """Run one design shard IN-PROCESS on this worker's persistent device.

    The BoltzGen single-device pipeline runs each stage in-process on
    get_device(), so invoking it here transparently reuses this worker's
    already-open chip — no per-shard cold-open, hence no UMD device-init deadlock.
    A daemon thread relays the pipeline's stage progress while the run blocks this
    (heartbeat-covered) worker thread; cancellation is shard-granular (the run
    finishes its shard, bounded by the platform watchdog)."""
    import threading
    job_id = job["id"]
    t0 = time.time()
    device = str(worker_info["device_id"])

    def emit(event: str, **kw):
        try:
            client.event(run_id, worker_id, {"event": event, **meta, **kw})
        except Exception:
            pass

    workdir = Path(tempfile.mkdtemp(prefix=f"tt-bio-design-{job_id}-"))
    out_dir = workdir / "out"
    progress_file = workdir / "progress.jsonl"
    progress_file.write_text("")
    row: dict[str, Any] = {"id": job_id, "status": "failed"}
    outputs: dict[str, str] = {}
    emit("start", name=job_id)

    # Relay BoltzGen's per-stage progress while run_command blocks this thread
    # (heartbeats keep flowing on the background thread set up in run_worker_loop).
    pos = [0]
    stop = threading.Event()

    def _pump():
        while not stop.wait(1.0):
            pos[0] = _forward_design_progress(progress_file, pos[0], emit)

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()
    try:
        data = json.loads(base64.b64decode(job.get("input_b64", "")).decode("utf-8"))
        num_designs = int(data.get("num_designs") or 1)
        spec_paths = []
        for spec in cfg.get("specs", []):
            p = workdir / Path(str(spec["name"])).name
            p.write_text(str(spec["content"]))
            spec_paths.append(str(p))
        if not spec_paths:
            raise RuntimeError("design run has no spec")

        # execute_command reads this per-call, so set it just before running.
        os.environ["BOLTZGEN_PROGRESS_FILE"] = str(progress_file)
        argv = ["run", *spec_paths, "--output", str(out_dir),
                "--num_designs", str(num_designs), "--device_ids", device,
                "--protocol", cfg.get("protocol", "protein-anything")]
        if cfg.get("steps"):
            argv += ["--steps", *cfg["steps"]]
        if cfg.get("fast"):
            argv.append("--fast")
        if cfg.get("moldir"):
            argv += ["--moldir", str(cfg["moldir"])]

        from tt_bio.boltzgen.cli.boltzgen import build_parser, run_command
        run_command(build_parser().parse_args(argv))  # reuses get_device(); no cold-open
        outputs = _read_outputs(out_dir)
        row.update({"status": "ok", "num_designs": num_designs,
                    "runtime_s": round(time.time() - t0, 1)})
    except Exception as exc:
        traceback.print_exc()
        row["error"] = str(exc)[:200]
    finally:
        stop.set()
        pos[0] = _forward_design_progress(progress_file, pos[0], emit)  # flush the tail
        shutil.rmtree(workdir, ignore_errors=True)
        os.environ.pop("BOLTZGEN_PROGRESS_FILE", None)
        gc.collect()  # drop the design models' host refs; the chip stays open

    try:
        client.complete(
            run_id, worker_id, row,
            {**meta, "event": "done", "name": job_id, "status": row["status"],
             "time": round(time.time() - t0, 1), "error": row.get("error", ""), "row": row},
            outputs=outputs or None,
        )
    except Exception:
        traceback.print_exc()


def _forward_design_progress(path: Path, pos: int, emit) -> int:
    """Tail BoltzGen's progress JSONL and relay each stage start to the
    controller as a 'stage' event, so the orchestrator (and the platform's
    progress bar) see the pipeline advance live. Returns the new read offset."""
    try:
        with open(path, "r") as f:
            f.seek(pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("t") == "start" and ev.get("name"):
                    emit("stage", stage=ev["name"])
            return f.tell()
    except Exception:
        return pos


def _read_outputs(output_dir: Path) -> dict[str, str]:
    """Read every file in output_dir and return name -> base64 bytes."""
    outputs: dict[str, str] = {}
    if not output_dir.exists():
        return outputs
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        outputs[rel] = base64.b64encode(path.read_bytes()).decode("ascii")
    return outputs


def _complete_failure(
    client: ControllerClient,
    run_id: str,
    worker_id: str,
    meta: dict[str, Any],
    jobs: list[dict[str, Any]],
    error: str,
) -> None:
    """Mark each leased job as failed when worker setup itself fails."""
    for job in jobs:
        row = {"id": job["id"], "status": "failed", "error": error}
        try:
            client.complete(
                run_id,
                worker_id,
                row,
                {**meta, "event": "done", "name": job["id"], "status": "failed",
                 "time": 0, "error": error, "row": row},
            )
        except Exception:
            pass

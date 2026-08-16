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

from tt_bio.device_lease import install_parent_death_guard
from tt_bio.distributed import ControllerClient, HttpProgressQueue


_REAL_STDERR_FD: int | None = None


def _silence_subprocess_output() -> None:
    """Send stdout/stderr to /dev/null so kernel/library noise stays hidden.

    Keeps one dup of the real stderr. A worker that dies before it has a run to
    attach an event to has no other way to say why, and writing that fatal to
    /dev/null is what left a device-open failure invisible for hours: the CLI
    hung, the log was 0 bytes, and the traceback had already been discarded.
    """
    global _REAL_STDERR_FD
    _REAL_STDERR_FD = os.dup(2)
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    dn_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn_fd, 1)
    os.dup2(dn_fd, 2)
    os.close(dn_fd)


def _report_fatal(message: str) -> None:
    """Write a worker-fatal to the launcher's real stderr, silenced or not."""
    fd = _REAL_STDERR_FD if _REAL_STDERR_FD is not None else 2
    try:
        os.write(fd, message.encode("utf-8", "replace"))
    except Exception:
        pass


def _apply_tt_environment(worker_info: dict[str, Any]) -> None:
    """Configure TT visibility for this worker before importing ttnn."""
    if worker_info["accelerator"] != "tenstorrent":
        return
    os.environ["TT_VISIBLE_DEVICES"] = str(worker_info.get("visible_devices") or worker_info["device_id"])
    os.environ["TT_BIO_LOGICAL_DEVICE_ID"] = str(worker_info.get("logical_device_id", 0))
    mgd = worker_info.get("mesh_graph_descriptor")
    if mgd and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)


def _bind_host_threads() -> None:
    """Bind torch's thread pools to the cap ``main._cap_worker_threads`` exported
    before spawning. See ``runtime.bind_host_threads``."""
    from . import runtime

    runtime.bind_host_threads()


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
    # OpenFold3: checkpoint via $OF3_CKPT or the local cache (no tt-bio HF mirror;
    # the p2 preview weights are distributed by the OpenFold consortium). MSA dir
    # resolves exactly like Protenix-v2.
    if cfg.get("model") == "openfold3":
        cfg["msa_dir"] = _resolve_msa_dir(cfg.get("msa_dir"), cache)
        of3_ckpt = os.environ.get("OF3_CKPT") or str(cache / "of3-p2-155k.pt")
        if not Path(of3_ckpt).exists():
            raise FileNotFoundError(
                "OpenFold3 checkpoint not found. Set OF3_CKPT to the OF3 p2 preview "
                "weights file (of3-p2-155k.pt) or place it at "
                f"{cache / 'of3-p2-155k.pt'}.")
        cfg["of3_ckpt"] = of3_ckpt
        tmpl_struct_dir = Path(
            os.environ.get("OF3_TEMPLATE_STRUCTURES")
            or str(cache / "of3_template_structures"))
        tmpl_struct_dir.mkdir(parents=True, exist_ok=True)
        cfg["of3_template_structures"] = str(tmpl_struct_dir)
        cfg["of3_max_msa_seqs"] = os.environ.get("OF3_MAX_MSA_SEQS")
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
    # Embedding models (ESMC, SaProt): weights come straight from the HF cache,
    # no Boltz-2 checkpoints/molecule library/MSA dir needed.
    if _is_embed_model(cfg.get("model", "boltz2")):
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


def _write_openfold3_structure(atom_array, coords, outpath, output_format,
                               b_factors=None):
    """Write an OpenFold3 prediction as PDB/mmCIF: the featurization's biotite
    AtomArray carries all atom metadata; only the coordinates and the per-atom
    pLDDT B-factors (0-100, the AF/Boltz convention) are replaced."""
    import biotite.structure.io.pdb as _pdb
    import biotite.structure.io.pdbx as _pdbx

    arr = atom_array.copy()
    arr.coord = coords.detach().cpu().numpy().astype("float32")
    if "b_factor" not in arr.get_annotation_categories():
        arr.add_annotation("b_factor", float)
    arr.b_factor[:] = (b_factors.detach().cpu().numpy().astype("float32")
                       if b_factors is not None else 0.0)
    # Bio.PDB.MMCIFParser hard-requires _atom_site.occupancy; biotite only writes
    # it when the annotation exists.
    if "occupancy" not in arr.get_annotation_categories():
        arr.add_annotation("occupancy", float)
        arr.occupancy[:] = 1.0
    outpath = Path(outpath)
    if output_format == "pdb":
        pf = _pdb.PDBFile()
        pf.set_structure(arr)
        pf.write(str(outpath))
    else:
        cf = _pdbx.CIFFile()
        _pdbx.set_structure(cf, arr)
        cf.write(str(outpath))


def _openfold3_template_map(path: Path) -> dict[str, str]:
    """Per-chain template alignment (npz) paths from a YAML input's `templates:` key.

    OF3-only: the shared chain reader has no template field, so the OF3 path re-reads
    the YAML for `{protein: {id: X, sequence: ..., templates: <npz>}}`. A template path
    on a non-protein chain, an unknown chain id, or a missing file is a hard error —
    silently dropping a user-supplied template would fold a different input than asked.
    """
    if path.suffix.lower() not in (".yml", ".yaml"):
        return {}
    import yaml
    doc = yaml.safe_load(path.read_text()) or {}
    out: dict[str, str] = {}
    for entry in doc.get("sequences", []):
        if not isinstance(entry, dict):
            continue
        for key, mt in (("protein", "protein"), ("rna", "rna"), ("dna", "dna")):
            sub = entry.get(key)
            if not (isinstance(sub, dict) and sub.get("sequence")):
                continue
            tmpl = sub.get("templates")
            if tmpl in (None, "", "~"):
                continue
            if mt != "protein":
                raise RuntimeError(
                    f"--model openfold3: templates are only valid on protein chains, "
                    f"not {mt} (chain entry {sub.get('id')!r}).")
            tp = Path(str(tmpl)).expanduser()
            if not tp.exists():
                raise RuntimeError(
                    f"--model openfold3: template file {tp} does not exist.")
            ids = sub.get("id", "A")
            id_list = ([str(x) for x in ids] if isinstance(ids, (list, tuple))
                       else str(ids).split(","))
            for c in id_list:
                out[c.strip()] = str(tp)
    return out


def _validate_openfold3_constraints(path) -> None:
    """Reject yaml `constraints:` blocks for OF3 instead of folding without them.

    The OF3 query is built with `covalent_bonds: None` (the bond graph is not
    ported), so a constraints block would otherwise be silently dropped — the
    silent-garbage class. Raises naming the constraint count and the models that
    do honor covalent bonds.
    """
    from tt_bio.main import _read_bio_constraints

    bonds = _read_bio_constraints(path)
    if bonds:
        raise RuntimeError(
            "--model openfold3 does not port covalent bonds yet "
            f"(got {len(bonds)} constraint(s) from {path.name}); the fold would "
            "silently ignore them. Remove the constraints block or use "
            "--model protenix-v2 / opendde.")


def _validate_openfold3_chains(chains: list) -> None:
    """Reject OF3 inputs that would otherwise fold into plausible-looking garbage.

    Ligand chains are out of scope (polymer-only port); a blank/whitespace sequence
    would tokenize to UNK placeholders and still produce a status=ok structure — the
    silent-garbage class from tt-bio-fold-succeeds-on-malformed-input. Unknown residue
    CODES (X/Z/...) stay upstream-compatible: the vendored featurizer maps them to UNK
    with a warning, exactly like the reference implementation.
    """
    if not chains:
        raise RuntimeError("no protein/nucleic-acid sequences")
    non_polymer = [cid for cid, _s, _sp, mt in chains if mt not in ("protein", "rna", "dna")]
    if non_polymer:
        raise RuntimeError(
            f"--model openfold3 is polymer-only for now (chain(s) {non_polymer} are "
            "ligands); see docs/openfold3-port.md.")
    blank = [cid for cid, cseq, _sp, _mt in chains if not cseq or not cseq.strip()]
    if blank:
        raise RuntimeError(
            f"--model openfold3: chain(s) {blank} have empty/whitespace-only sequences.")


def _prefetch_openfold3_template_structures(tmpl_map: dict[str, str],
                                            struct_dir: Path) -> None:
    # Download the raw template CIFs a `templates:` npz needs from RCSB. The npz
    # holds alignments only (index/release_date/idx_map per entry); coordinates
    # come from <pdb_id>.cif files. Only missing files are fetched; a failed
    # download is a hard error (a skipped template would fold the wrong input).
    import numpy as np
    pdb_ids: set[str] = set()
    for npz_path in tmpl_map.values():
        with np.load(npz_path, allow_pickle=True) as z:
            pdb_ids |= {k.split("_")[0] for k in z.keys()}
    missing = [p for p in sorted(pdb_ids)
               if not (struct_dir / f"{p}.cif").exists()]
    if not missing:
        return
    import urllib.request
    for p in missing:
        url = f"https://files.rcsb.org/download/{p.upper()}.cif"
        try:
            urllib.request.urlretrieve(url, struct_dir / f"{p}.cif")
        except Exception as exc:
            raise RuntimeError(
                f"--model openfold3: failed to fetch template structure {url}: {exc}")


def _err_text(exc: BaseException, limit: int = 2000) -> str:
    """Bounded error text for a job row that never loses the tail.

    TT_FATAL messages put the diagnostic payload LAST. An allocator OOM reads
    `TT_FATAL @ ...bank_manager.cpp:439 ... Out of Memory: Not enough space to
    allocate N B DRAM buffer across 12 banks, where each bank needs to store N B,
    but bank size is N B (allocated: N B, free: N B, largest free block: N B)` --
    the leading 78 chars are a fixed file/line prefix and the closing parenthetical
    is the only part that says how full the chip actually was. A plain
    `str(exc)[:200]` lands mid-number just before it, so every recorded OOM in the
    AbAg-XM Wormhole campaign was indistinguishable between a genuinely full chip
    and one oversized request. Keep both ends instead of just the head.

    400 was still too tight, for the opposite reason. A TT_THROW puts its payload in the
    MIDDLE -- `... clash with L1 buffers. L1 buffer allocated at X and static circular
    buffer region ends at Y`, followed by the backtrace frames that name the op -- so
    keeping both ends elided the diagnosis and the op together, and every L1
    circular-buffer clash in the AbAg-XM campaign was recorded unattributably. 2000 keeps
    the OOM parenthetical, the TT_THROW payload and the first backtrace frames.
    """
    s = str(exc)
    if len(s) <= limit:
        return s
    keep = limit - 5                       # room for the " ... " elision marker
    head = keep // 2
    return s[:head] + " ... " + s[-(keep - head):]


def _is_esmc_model(model_id: str) -> bool:
    """True for any ESMC embedding model name (esmc-300m/600m/6b).

    Lazily imports tt_bio.esmc (which imports ttnn at module scope) so a
    worker that never handles embed jobs never pays that import cost.
    """
    from tt_bio.esmc import MODELS

    return model_id in MODELS


def _is_saprot_model(model_id: str) -> bool:
    """True for any SaProt embedding model name (saprot-35m/650m/1.3b)."""
    from tt_bio.saprot import MODELS

    return model_id in MODELS


def _is_embed_model(model_id: str) -> bool:
    """True for any model this worker serves through the embed path.

    Both families produce ``ESMCEmbedding`` and ship through the same npz/parquet
    writer; they differ only in loader and tokenizer. Dispatching on the family
    rather than on ESMC alone is what lets a saprot-* job reach a worker at all --
    before this, it fell through to the Boltz-2 branch and tried to load Boltz-2
    checkpoints.
    """
    return _is_esmc_model(model_id) or _is_saprot_model(model_id)


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
        elif model_id == "openfold3":
            import ttnn

            from tt_bio.openfold3_fold import OpenFold3
            from tt_bio.tenstorrent import get_device

            dev = get_device()
            ckc = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True, packer_l1_acc=True)
            sd = torch.load(cfg["of3_ckpt"], map_location="cpu", weights_only=False)
            # CLI --recycling_steps counts recycles; the trunk runs recycles+1 cycles.
            self.model = OpenFold3(sd, ckc, num_cycles=int(cfg.get("recycling_steps") or 3) + 1)
        elif model_id in ("opendde", "opendde-abag"):
            from tt_bio.opendde import OpenDDE

            self.model = OpenDDE.load_from_checkpoint(
                cfg.get("opendde_ckpt"), abag=(model_id == "opendde-abag"))
        elif _is_esmc_model(model_id):
            from tt_bio.esmc import load_esmc

            if model_id in ("esmc-300m", "esmc-600m"):
                # The traced single-sequence forward needs its trace region
                # reserved at device-open; load_esmc requests it only on a
                # fresh open. Guarantee the fresh open here instead of relying
                # on reset() having closed the chip — a cleanup that failed
                # (swallowed there) would otherwise pin this worker eager
                # forever. Scoped to the 300M/600M embedders: esmc-6b cannot
                # be traced, and a 256 MB reservation risks OOM for the big
                # models, so no other model's device-open is touched.
                from tt_bio.tenstorrent import cleanup as _tt_cleanup
                from tt_bio.tenstorrent import trace_region_size

                if trace_region_size() <= 0:
                    _tt_cleanup()
            self.model = load_esmc(model_id, fast=cfg.get("fast", False))
        elif _is_saprot_model(model_id):
            from tt_bio.saprot import load_saprot

            # No trace region: SaProt has no traced forward at any batch size.
            self.model = load_saprot(model_id, fast=cfg.get("fast", False))
        else:
            from tt_bio.boltz2 import Boltz2
            from tt_bio.data.featurizer import Boltz2Featurizer
            from tt_bio.data.mol import load_canonicals
            from tt_bio.data.tokenize import Boltz2Tokenizer

            self._tokenizer, self._featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
            self._mol_dir = Path(cfg["mol_dir"])
            self._ccd = load_canonicals(self._mol_dir)
            from tt_bio.tenstorrent import diffusion_fp32_device

            struct_fp32_device = (
                os.environ.get("BOLTZ2_STRUCTURE_DIFFUSION_FP32_DEVICE", "0") == "1"
            )
            with diffusion_fp32_device(struct_fp32_device):
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
        every forward the device runs in bf16 — the structure ``predict_step`` (bf16 unless
        BOLTZ2_STRUCTURE_DIFFUSION_FP32_DEVICE=1) AND the affinity ``aff_model.predict_step``
        (bf16 unless BOLTZ2_AFFINITY_DIFFUSION_FP32_DEVICE=1) — so the bf16 reference mirrors
        the device's dtype boundary rather than leaving the scalar in fp32. Shared draws are
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
        if cfg.get("model") == "openfold3":
            return self._predict_openfold3_one(path, cfg)
        if cfg.get("model", "boltz2") in ("esmfold2", "esmfold2-fast"):
            return self._predict_esmfold2_one(path, cfg)
        if _is_embed_model(cfg.get("model", "boltz2")):
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
            for _cid, seq, spec, _mods in chains:
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
        chains = [(cid, seq, resolve_msa(spec, seq, msa_dir, max_sequences=max_msa) if uses_msa else None, mods)
                  for cid, seq, spec, mods in chains]
        ranked = fold_complex(
            self.model, chains,
            num_loops=cfg["recycling_steps"], num_sampling_steps=cfg["sampling_steps"],
            num_diffusion_samples=cfg["diffusion_samples"], seed=cfg.get("seed") or 0,
            return_all=True,
        )
        res = ranked[0]
        # Write every sample, not just the winner: best as "{stem}.{fmt}" and the rest as
        # "{stem}_model_{rank}.{fmt}", the same convention Protenix-v2 and OpenDDE use. The
        # samples were always drawn -- they were discarded in fold_complex -- so this costs
        # nothing but the writes, and the winner's file is byte-identical to before.
        fmt = cfg["output_format"]
        struct_dir = Path(cfg["struct_dir"])
        for r, s in enumerate(ranked):
            name = f"{path.stem}.{fmt}" if r == 0 else f"{path.stem}_model_{r}.{fmt}"
            _write_structure(s.complex, struct_dir / name, fmt)

        def _sample_scalars(s):
            m = {"plddt": round(float(s.plddt.mean()), 4)}
            if getattr(s, "ptm", None) is not None:
                m["ptm"] = round(float(s.ptm), 4)
            return m

        metrics = {
            **_sample_scalars(res),
            "n_residues": sum(len(c[1]) for c in chains), "n_chains": len(chains),
            "msa": any(c[2] is not None for c in chains),
            "samples": cfg["diffusion_samples"],  # best-of-N: report N (plddt is the winner's)
        }
        # Per-sample records, keyed like the other models' so one dataset harness reads them all.
        # Only when there is more than one, matching _scalars/all_runs in main.py.
        if len(ranked) > 1:
            metrics["all_runs"] = [{"rank": r, **_sample_scalars(s)} for r, s in enumerate(ranked)]
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

        # --msa_cache_only: the cache is the only source, so a miss is an error. Without this
        # _resolve_a3m_text returns None for an uncached chain and the fold quietly proceeds
        # single-sequence for it -- a large, invisible accuracy loss in a benchmark run.
        if cfg.get("msa_cache_only"):
            uncached = [cid for (cid, _s, _sp, mt), (_q, a3m, _m)
                        in zip(chains, chain_specs) if mt == "protein" and not a3m]
            if uncached:
                raise RuntimeError(
                    f"--msa_cache_only: no cached a3m in {msa_dir} for protein chain(s) "
                    f"{uncached}. Folding them single-sequence would silently change MSA "
                    "depth; search them first, or drop --msa_cache_only.")

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
                return_confidence=True, max_parallel_samples=cfg.get("max_parallel_samples"))
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

    def _protenix_inputs(self, path: Path, cfg: dict[str, Any]):
        """Sequences -> (optional per-chain MSA) -> model-ready features for one target.
        Shared by the single and batched protenix entry points."""
        import hashlib

        from tt_bio.esmfold2 import report_progress
        from tt_bio.main import (_generate_esmfold2_a3m, _read_bio_chains,
                                 _read_bio_constraints, _resolve_a3m_text)
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
        return feats, chains, chain_specs

    def _protenix_emit(self, path: Path, cfg: dict[str, Any], feats, chains, chain_specs,
                       coords, confs):
        """Rank samples, write structures, build the metrics row for one target."""
        import types

        from tt_bio.main import _write_protenix_structure

        # AF-style ranking score: ipTM-weighted for complexes, pTM for monomers,
        # falling back to pLDDT only if neither is available. Picks the best sample
        # and orders all_runs -- mirrors Boltz-2's confidence_score ranking.
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
            "n_atoms": int(coords[0].shape[-2]), "samples": len(confs),
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
        from tt_bio.esmfold2 import report_progress

        feats, chains, chain_specs = self._protenix_inputs(path, cfg)

        # One shared progress path: report_progress has exactly the progress_fn
        # signature, so hand it straight to the model -- trunk iterations report
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
                max_parallel_samples=cfg.get("max_parallel_samples"),
                # Without this --trace was a silent no-op for --model protenix-v2: main.py
                # reserves the trace region and puts "trace" in the worker config, and
                # Protenix.fold accepts trace=, but this call site never forwarded it, so
                # every protenix fold ran the untraced per-step dispatch. The OpenDDE branch
                # above forwards it, which is why the flag looked plumbed.
                trace=cfg.get("trace", False),
            )
        confs = conf if isinstance(conf, list) else [conf]
        return self._protenix_emit(path, cfg, feats, chains, chain_specs, coords, confs)

    def predict_many(self, paths: list, cfg: dict[str, Any]):
        """Fold several targets in one batched diffusion trajectory. Returns one
        (metrics, best, feats) tuple per input path, in input order.

        Only protenix-v2 has a batched path today; anything else folds serially through
        predict_one, so callers can always use this entry point. Targets must share their
        atom and token counts (bucket first) -- a mismatch raises.
        """
        if cfg.get("model") != "protenix-v2" or len(paths) == 1 or int(cfg["diffusion_samples"]) != 1:
            return [self.predict_one(p, cfg) for p in paths]

        from tt_bio.esmfold2 import report_progress

        prepared = [self._protenix_inputs(p, cfg) for p in paths]
        with torch.no_grad(), self._maybe_ref_bf16():
            coords, confs = self.model.fold_many(
                [f for f, _c, _s in prepared], n_step=cfg["sampling_steps"],
                seed=cfg.get("seed") or 0, progress_fn=report_progress,
                return_confidence=True, n_cycles=cfg.get("recycling_steps"))
        return [self._protenix_emit(p, cfg, prepared[b][0], prepared[b][1], prepared[b][2],
                                    coords[b], [confs[b]])
                for b, p in enumerate(paths)]


    def _predict_openfold3_one(self, path: Path, cfg: dict[str, Any]):
        """OpenFold3 fold: sequence(s) -> per-chain MSA -> fixture-free on-device fold
        (host featurization + device glue/trunk/diffusion/confidence) -> structure.
        Rides the same MSA stage as Protenix-v2: uncached protein chains are searched
        into the shared msa_dir and attached to the query as main_msa_file_paths.
        Polymer chains only (protein/RNA/DNA). Templates are opt-in per protein chain
        via the YAML `templates:` key (precomputed alignment npz, the format the
        upstream benchmark cache ships); there is no template SEARCH."""
        import types

        from tt_bio.esmfold2 import report_progress
        from tt_bio.main import _read_bio_chains, _read_bio_constraints

        chains = _read_bio_chains(path)
        _validate_openfold3_chains(chains)
        _validate_openfold3_constraints(path)
        tmpl_map = _openfold3_template_map(path)
        unknown_tmpl = sorted(set(tmpl_map) - {cid for cid, _s, _sp, _mt in chains})
        if unknown_tmpl:
            raise RuntimeError(
                f"--model openfold3: `templates:` given for unknown chain id(s) "
                f"{unknown_tmpl}.")
        msa_dir = Path(cfg["msa_dir"])

        report_progress("msa")
        _MT = {"protein": "PROTEIN", "rna": "RNA", "dna": "DNA"}
        query = {
            "query_name": path.stem, "use_msas": True, "use_paired_msas": False,
            "use_main_msas": True, "covalent_bonds": None,
            "chains": [
                {"molecule_type": _MT[mt], "chain_ids": [cid], "sequence": cseq,
                 "non_canonical_residues": None, "smiles": None, "ccd_codes": None,
                 "paired_msa_file_paths": None,
                 "main_msa_file_paths": ([str(Path(spec).expanduser())]
                                         if spec and Path(spec).expanduser().exists()
                                         else None),
                 "template_alignment_file_path": tmpl_map.get(cid),
                 "template_entry_chain_ids": None,
                 "sdf_file_path": None}
                for cid, cseq, spec, mt in chains
            ],
        }

        report_progress("prep")
        import json as _json
        import tempfile

        import numpy as np
        import torch as _torch

        from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
            InferenceQuerySet,
        )
        from tt_bio.openfold3_data import (
            build_openfold3_features, make_openfold3_msa_features)
        from tt_bio.openfold3_host_prep import (
            dedup_template_slots, derive_block_aux, derive_relpos,
            derive_template_feat, ref_atom_embed, run_input_atom_encoder)
        from tt_bio.openfold3_weights import _sub

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            _json.dump({"queries": {path.stem: query}}, fh)
            qpath = fh.name
        seed = int(cfg.get("seed") or 0)
        _torch.manual_seed(0)
        np.random.seed(0)
        # The vendored featurizer draws its RDKit conformer seed from python"s
        # unseeded `random` (conformer.py: "we set a random seed here"), which made
        # ref_pos — and therefore the whole fold — nondeterministic across
        # processes even at fixed --seed. Pin it to the fold seed.
        import random as _pyrandom
        _pyrandom.seed(seed)
        iqs = InferenceQuerySet.from_json(qpath)
        of3_query = next(iter(iqs.queries.values()))
        # The MSA stage delegates to the shared resolver: it exposes the cached
        # hash-named ColabFold a3m under the canonical source basename OF3's parser
        # filters on (a raw hash-named path parses to ZERO chains and dies on an
        # IndexError deep in the vendored pipeline), and preserves user-specified
        # per-chain MSA paths.
        want_msa = cfg.get("use_msa_server") or cfg.get("msa_db_path") or cfg.get("msa_endpoint")
        from tt_bio.openfold3_data import resolve_openfold3_msas
        of3_query = resolve_openfold3_msas(
            of3_query, msa_dir, target_id=path.stem,
            msa_db_path=cfg.get("msa_db_path"),
            use_envdb=cfg.get("use_envdb", False),
            msa_server_url=cfg.get("msa_server_url"),
            msa_pairing_strategy=cfg.get("msa_pairing_strategy"),
            msa_server_username=cfg.get("msa_server_username"),
            msa_server_password=cfg.get("msa_server_password"),
            api_key=cfg.get("api_key_value"),
            msa_endpoint=cfg.get("msa_endpoint"),
            fetch=bool(want_msa))
        if want_msa:
            missing = [c.chain_ids for c in of3_query.chains
                       if c.molecule_type.name == "PROTEIN" and not c.main_msa_file_paths]
            if missing:
                raise RuntimeError(
                    f"MSA was requested but none resolved for protein chain(s) {missing} "
                    "-- refusing to silently fold single-sequence.")
        if not any(c.main_msa_file_paths for c in of3_query.chains):
            of3_query.use_msas = False
            of3_query.use_main_msas = False
        if tmpl_map:
            _prefetch_openfold3_template_structures(
                tmpl_map, Path(cfg["of3_template_structures"]))
        features = build_openfold3_features(
            of3_query,
            template_structures_directory=cfg["of3_template_structures"])
        # Default = the featurizer max_rows (16384), i.e. NO extra subsampling: the
        # CPU reference folds the full featurized MSA, so any lower cap is an input
        # divergence (measured on 9BK6: the 1024-row subsample cost chain A
        # 11.1 vs 7.6 A Ca-RMSD). OF3_MAX_MSA_SEQS stays as a memory escape hatch.
        msa_feat = make_openfold3_msa_features(
            features, max_sequences=int(cfg.get("of3_max_msa_seqs") or 16384), seed=0)
        aux = derive_block_aux(features)
        template_feat, template_slots = dedup_template_slots(
            derive_template_feat(features))
        relpos = derive_relpos(features)

        model = self.model
        dev = model.device
        ai = run_input_atom_encoder(dev, model.ckc, model.sd, features, aux)
        s_input = _torch.cat(
            [ai, features["restype"], features["profile"],
             features["deletion_mean"].unsqueeze(-1)], dim=-1)
        cl0, plm0 = ref_atom_embed(
            _sub(model.sd,
                 "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), features)
        dm_aux_host = dict(
            cl0=cl0, plm0=plm0, atom_mask=aux["atom_mask"],
            atom_to_token_index=aux["atom_to_token_index"],
            npe_q_indices=aux["npe_q_indices"], npe_k_indices=aux["npe_k_indices"],
            zij_mask=aux["zij_mask"], key_block_idxs=aux["key_block_idxs"],
            invalid_mask=aux["invalid_mask"], mask_trunked=aux["mask_trunked"],
            atom_to_token_mean=aux["atom_to_token_mean"], nb=aux["nb"], NP=aux["NP"])
        ca_mask = aux["ca_mask"]
        atom_to_token = aux["atom_to_token_index"].long()
        polymer_token = (features["is_protein"] | features["is_rna"]
                         | features["is_dna"]).bool()
        confidence_aux = dict(
            representative_atom_indices=_torch.from_numpy(
                np.flatnonzero(ca_mask.numpy())).long(),
            max_atom_per_token_mask=aux["max_atom_per_token_mask"],
            atom_array=features["atom_array"], asym_id=features["asym_id"],
            atom_to_token_index=atom_to_token, atom_mask=features["atom_mask"].bool(),
            polymer_mask=polymer_token[atom_to_token],
            repr_batch={k: features[k] for k in (
                "is_protein", "is_dna", "is_rna", "is_atomized", "restype",
                "start_atom_index", "atom_mask", "token_mask")})

        n_sample = int(cfg["diffusion_samples"])
        result = model.fold(
            template_feat=template_feat, template_slots=template_slots,
            msa_feat=msa_feat, s_input=s_input,
            relpos=relpos, token_bonds=features["token_bonds"],
            token_mask=features["token_mask"], dm_aux_host=dm_aux_host,
            n_atom=aux["n_atom"], n_token=aux["n_token"],
            no_rollout_steps=int(cfg["sampling_steps"]), seed=seed,
            no_samples=n_sample, confidence_aux_host=confidence_aux,
            progress_fn=report_progress)

        confs = result.confidence
        order = sorted(range(len(confs)),
                       key=lambda k: confs[k]["ranking_score"], reverse=True)
        rank_of = {k: r for r, k in enumerate(order)}

        struct_dir = Path(cfg["struct_dir"])
        stem, fmt = path.stem, cfg["output_format"]
        for k in range(len(confs)):
            r = rank_of[k]
            name = f"{stem}.{fmt}" if r == 0 else f"{stem}_model_{r}.{fmt}"
            _write_openfold3_structure(
                features["atom_array"], result.samples[k], struct_dir / name, fmt,
                b_factors=confs[k]["plddt_atom"] * 100.0)

        def _row(c):
            return {"complex_plddt": round(c["plddt"], 6), "plddt": round(c["plddt"], 6),
                    "ptm": round(c.get("ptm", 0.0), 6), "iptm": round(c.get("iptm", 0.0), 6),
                    "confidence_score": round(c["ranking_score"], 6)}

        best = confs[order[0]]
        metrics = {
            **_row(best),
            "n_residues": sum(len(cseq) for _c, cseq, _s, _mt in chains),
            "n_chains": len(chains), "n_tokens": int(features["restype"].shape[0]),
            "msa": any(c.main_msa_file_paths for c in of3_query.chains),
            "n_atoms": int(result.samples[0].shape[0]), "samples": n_sample,
        }
        if len(confs) > 1:
            metrics["all_runs"] = [{"rank": rank_of[k], **_row(confs[k])} for k in order]
        return metrics, None, {"record": types.SimpleNamespace(affinity=False)}

    def _predict_embed_one(self, path: Path, cfg: dict[str, Any]):
        """Embed one job's shard of sequences with the resident embedding model.

        Serves both families: ESMC reads a ``{id: sequence}`` shard, SaProt a
        ``{id: [aa, 3di]}`` one.

        ``path`` is a YAML ``{id: sequence}`` mapping (one shard of a larger
        --controller embed run). Writes per-sequence ``.npz`` (or one shard
        parquet, named by job id to avoid colliding with other shards' output
        once every job's outputs land in the same directory) into struct_dir —
        the same output-shipping path predict/design jobs already use.
        """
        import types

        from tt_bio.esmc import write_npz_many, write_parquet

        model_id = cfg.get("model", "")
        if _is_saprot_model(model_id):
            from tt_bio.saprot import embed_sequences, read_shard_yaml

            sequences = read_shard_yaml(path)
        else:
            from tt_bio.esmc import embed_sequences, load_sequences

            sequences = load_sequences(path)
        t0 = time.perf_counter()
        results = embed_sequences(
            self.model, sequences, return_logits=cfg.get("return_logits", False),
            pool=cfg.get("pool", "mean"), batch_size=cfg.get("batch_size", 8),
        )
        device_s = time.perf_counter() - t0
        struct_dir = Path(cfg["struct_dir"])
        t0 = time.perf_counter()
        if cfg.get("output_format") == "parquet":
            write_parquet(results, struct_dir / f"{cfg['job_id']}.parquet")
        else:
            # Threaded, not a serial write_npz loop. np.savez_compressed sits in
            # zlib's C compress loop, which releases the GIL, and on the served
            # batch-8 x 76 aa shape the serial loop costs 369 ms against 64 ms of
            # device work -- 85 % of the job's wall spent compressing while the
            # chip idles (measured on the Wormhole Galaxy, UMD 26).
            write_npz_many(results, struct_dir)
        write_s = time.perf_counter() - t0
        metrics = {
            "n_sequences": len(results),
            "d_model": int(results[0].pooled.shape[0]) if results else 0,
            "ids": [e.id for e in results],
            "lengths": [len(e.sequence) for e in results],
            # Reported separately so an ESM-C embed leg is comparable to SaProt's,
            # which times embed_sequences alone.
            "device_s": round(device_s, 4),
            "write_s": round(write_s, 4),
        }
        return metrics, None, {"record": types.SimpleNamespace(affinity=False)}

    def predict_affinity(self, path: Path, pred_structure, cfg: dict[str, Any]) -> dict[str, float]:
        from tt_bio.boltz2 import Boltz2
        from tt_bio.main import to_batch

        if self.aff_model is None:
            from tt_bio.tenstorrent import diffusion_fp32_device

            fp32_device = (
                os.environ.get("BOLTZ2_AFFINITY_DIFFUSION_FP32_DEVICE", "0") == "1"
            )
            with diffusion_fp32_device(fp32_device):
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


def _install_orphan_guard(dispatcher_pid: int) -> None:
    """Die with the dispatcher, whatever this worker is in the middle of.

    ``run_worker_loop``'s orphan check only runs between leases, so a worker
    orphaned DURING a job never reaches it and keeps its chip open and its card
    lease flocked forever. One was found holding /dev/tenstorrent/0 and card 3's
    lease for 6 h at 100% CPU, deferring every later job pinned to that card.
    Mechanism and the parent-THREAD caveat: ``install_parent_death_guard``.

    A worker can afford the SIGTERM handler ``_install_signal_handlers`` puts on
    top of this, which unwinds through close_device and leaves the chip clean,
    because ``run_worker_loop``'s heartbeat catches a worker too deep in ttnn to
    take the signal 60 s later. A process with no such backstop must leave SIGTERM
    at its default disposition instead, or a wedged chip is held forever.

    Workers are started from the dispatcher's main thread
    (``main._spawn_worker_processes``, ``_supervise_worker_processes``), which is
    what makes the kernel's parent-thread delivery rule a no-op here.
    """
    install_parent_death_guard(dispatcher_pid)


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
    _bind_host_threads()

    # A locally-spawned worker (CLI fan-out / serve pool) records its dispatcher
    # here; a dispatcher killed with SIGTERM skips its finally-block and would
    # otherwise orphan us holding the chip open indefinitely (observed: a stray
    # worker pinned /dev/tenstorrent/3 for 2h, silently blocking later runs on
    # that card). Remote `worker --connect` processes leave the var unset.
    _dispatcher_pid = int(os.environ.get("TT_BIO_PARENT_PID") or 0)
    _install_orphan_guard(_dispatcher_pid)

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
        orphaned_since = None
        while not _stop_beat.wait(8.0):
            if _dispatcher_pid and os.getppid() != _dispatcher_pid:
                # PDEATHSIG (see _install_orphan_guard) had 60 s to unwind this
                # cleanly and did not, so we are stuck inside a call that does not
                # take signals. Leave anyway: an orphan has no consumer for its
                # results, and its card lease blocks every later job on this chip.
                orphaned_since = orphaned_since or time.monotonic()
                if time.monotonic() - orphaned_since > 60:
                    os._exit(70)
                continue
            orphaned_since = None
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
            _report_fatal(f"tt-bio worker {worker_info['label']}: device open failed\n"
                          f"{traceback.format_exc()}")
            # The chip didn't come up with working local dispatch (e.g. a raced
            # "remote-only" bring-up). Do NOT stay online serving jobs we'd fail:
            # exit so the pool supervisor respawns us. The respawn reopens under the
            # host-wide device-init lock (one chip at a time), which is exactly what
            # clears the concurrent-init race behind a bad bring-up.
            return
    try:
        while True:
            if _dispatcher_pid and os.getppid() != _dispatcher_pid:
                return
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
                _complete_failure(client, run_id, worker_id, meta, jobs, _err_text(exc))
                state.reset()
                continue

            for job in jobs:
                _execute_job(state, job, cfg, run_id, client, worker_id, meta)
    except KeyboardInterrupt:
        pass
    except BaseException:
        # Anything escaping the per-job handling above kills this worker, and with
        # stdout/stderr on /dev/null the traceback would otherwise vanish — the
        # only diagnostic a local `predict` fan-out (no supervisor) will ever get
        # for why a worker slot went quiet.
        _report_fatal(f"tt-bio worker {worker_info['label']}: uncaught error, exiting\n"
                      f"{traceback.format_exc()}")
        raise
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
        outputs = _read_outputs(output_dir, _shared_outputs_dir(cfg))
    except Exception as exc:
        traceback.print_exc()
        row["error"] = _err_text(exc)
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
        outputs = _read_outputs(out_dir, _shared_outputs_dir(cfg))
        row.update({"status": "ok", "num_designs": num_designs,
                    "runtime_s": round(time.time() - t0, 1)})
    except Exception as exc:
        traceback.print_exc()
        row["error"] = _err_text(exc)
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


# Marks an output value as "the file is already where the client wants it" rather than
# base64 bytes. Only ever emitted after _shared_outputs_dir proves co-location.
SHARED_OUTPUT_PREFIX = "tt-bio-shared-path:"


def _shared_outputs_dir(cfg: dict[str, Any]) -> Path | None:
    """The client's own results directory, when this worker provably shares it.

    Routing bulk results back as base64 inside JSON costs more than the compute for
    embeddings: a 1024-sequence esmc run ships ~635 MB of per-residue arrays, inflated
    33% by base64, through one controller process while every card sits idle. When the
    worker and the client are the same filesystem the copy is pure waste.

    "Provably" matters. A worker on another machine can happily create the client's
    path and write into it, and the client would then find nothing -- so writability is
    not proof of sharing. The client leaves a nonce file in its results directory and
    names it here; seeing that exact file is what makes co-location certain.
    """
    share = cfg.get("shared_outputs")
    if not isinstance(share, dict):
        return None
    directory, token = share.get("dir"), share.get("token")
    if not directory or not token or os.sep in str(token):
        return None
    d = Path(directory)
    return d if (d / str(token)).is_file() else None


def _read_outputs(output_dir: Path, share_dir: Path | None = None) -> dict[str, str]:
    """Read every file in output_dir and return name -> base64 bytes.

    With ``share_dir`` set (see _shared_outputs_dir) each file is moved there instead
    and reported as a path, so the bytes never enter the controller. Any file that
    cannot be moved falls back to base64 individually, so a partial failure degrades
    to the old behaviour rather than losing an output.
    """
    outputs: dict[str, str] = {}
    if not output_dir.exists():
        return outputs
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if share_dir is not None:
            target = share_dir / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
                outputs[rel] = SHARED_OUTPUT_PREFIX + str(target)
                continue
            except Exception:
                pass  # fall through to base64 for this one file
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

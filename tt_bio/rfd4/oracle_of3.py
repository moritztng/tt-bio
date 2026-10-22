"""OpenFold3 behind the RFD4 oracle seam -- upstream's own second oracle backend.

Upstream's default scoring oracle is AlphaFold3 in a container; its second, shipped in
`configs/metrics/refolding_oracles/of3.yaml` and used by the released PPI search recipe, is the
same class on converted OpenFold3 weights. AF3's weights are not redistributable, so of3 is the
backend tt-bio can actually serve, and it is the one this adapter implements.

That substitution is a QUALITY question, not a free swap: whether an OF3-scored beam picks the
same designs an AF3-scored beam would is for `rfd4-quality` to grade, and until it has, no gated
number may assume they agree.

The fold runs through the serving path's own callables (`worker.openfold3_host_inputs` and
`openfold3_device_fold`) rather than a copy, so a change to how tt-bio folds reaches the oracle.
The confidence dict comes back in memory -- pae, plddt_atom, ptm, iptm, pair_chains_iptm -- so
scoring a candidate costs one fold and no file round trip.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tt_bio.rfd4.oracle import OracleRequest, extract_confidence_metrics, prefix, summary_confidences
from tt_bio.rfd4.oracle_af2ig import _pair_matrix, chain_sequence, residue_view


def structure_chains(structure_text: str) -> list[str]:
    """The chain ids a candidate carries, in the order its tokens appear.

    `tt-bio design --model rfd4` writes mmCIF, so the text arriving here is usually a CIF;
    `to_pdb_text` converts it, as `chain_sequence` beside it already did.
    """
    import io

    import biotite.structure.io.pdb as pdb

    from tt_bio.af2ig import to_pdb_text

    atoms = pdb.PDBFile.read(io.StringIO(to_pdb_text(structure_text))).get_structure(model=1)
    _first, chains = residue_view(atoms)
    seen: list[str] = []
    for chain in chains.tolist():
        if chain not in seen:
            seen.append(chain)
    return seen


def token_plddt(plddt_atom, atom_to_token_index, n_token: int) -> np.ndarray:
    """Per-token pLDDT, the mean over each token's atoms. OF3 reports pLDDT per ATOM; AF3's
    summary is per token, and a reward keyed on `plddt_coi` means the per-token one."""
    atom = np.asarray(plddt_atom, dtype=np.float64).reshape(-1)
    index = np.asarray(atom_to_token_index, dtype=np.int64).reshape(-1)
    total = np.bincount(index, weights=atom, minlength=n_token)
    count = np.bincount(index, minlength=n_token)
    return np.divide(total, np.maximum(count, 1), out=np.zeros(n_token), where=count > 0)


@dataclass
class OpenFold3Oracle:
    """One resident OpenFold3, folding candidate after candidate.

    `setup` opens the device and loads the weights once; `fold_batch` reuses both, which is the
    only way a beam search is affordable (`tt_bio/worker.py` keeps a model across folds for the
    same reason). `teardown` drops the weights and leaves the device open, because closing and
    reopening deadlocks UMD.
    """

    msa_dir: str
    template_structures: str
    name: str = "oracle"
    #: upstream's `oracle_name` for this key under `refolding_oracles: of3`; it names the
    #: refold-cache files evaluate looks for, while `name` prefixes the columns.
    oracle_name: str = "of3"
    model_id: str = "openfold3"
    #: upstream's AF3 oracle protocol: --num_recycles=5 --num_diffusion_samples=1
    recycles: int = 5
    sampling_steps: int = 200
    diffusion_samples: int = 1
    single_sequence: bool = True
    ckpt: str | None = None
    model: Any = None
    work: Path | None = field(default=None)

    def setup(self) -> None:
        if self.model is not None:
            return
        import ttnn

        from tt_bio.openfold3_fold import OpenFold3
        from tt_bio.tenstorrent import get_device
        from tt_bio.worker import _ensure_local_artifacts, _of3_state_dict

        dev = get_device()
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        # The checkpoint resolves the way a served fold resolves it: `_ensure_local_artifacts`
        # honours the env overrides, verifies the cache and re-fetches only what is missing, and
        # it is what fills the `of3_ckpt` key `_of3_state_dict` reads. Building the cfg by hand
        # left that key absent, so the oracle could not load weights at all.
        cfg = {"model": self.model_id}
        if self.ckpt:
            cfg["checkpoint"] = self.ckpt
        _ensure_local_artifacts(cfg)
        # The trunk runs recycles+1 cycles, as --recycling_steps counts recycles.
        self.model = OpenFold3(_of3_state_dict(cfg), ckc, num_cycles=int(self.recycles) + 1)

    def teardown(self) -> None:
        self.model = None

    def fold_batch(self, jobs: list[OracleRequest]) -> list[dict]:
        if self.model is None:
            raise RuntimeError("OpenFold3Oracle.setup() must run before fold_batch.")
        out = []
        for job in jobs:
            try:
                out.append(prefix(self.name, self.score_one(job)))
            except Exception:
                # A failed fold yields no keys, so the reward is -inf and the candidate is
                # dropped. Ranking it on a partial dict would be worse than losing it.
                out.append({})
        return out

    def _yaml(self, job: OracleRequest, chains: list[str]) -> Path:
        root = Path(self.work) if self.work else Path(tempfile.mkdtemp(prefix="of3_oracle_"))
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{job.example_id}.yaml"
        body = ["sequences:"]
        for chain in chains:
            body += ["  - protein:", f"      id: {chain}",
                     f"      sequence: {chain_sequence(job.structure, chain)}"]
        path.write_text("\n".join(body) + "\n")
        return path

    def score_one(self, job: OracleRequest) -> dict:
        from tt_bio.worker import openfold3_device_fold, openfold3_host_inputs

        chains = structure_chains(job.structure)
        cfg = {"model": self.model_id, "msa_dir": self.msa_dir,
               "template_structures": self.template_structures, "seed": 0,
               "single_sequence": bool(self.single_sequence), "output_format": "cif"}
        Path(self.msa_dir).mkdir(parents=True, exist_ok=True)
        Path(self.template_structures).mkdir(parents=True, exist_ok=True)
        host = openfold3_host_inputs(self._yaml(job, chains), cfg, model_name=self.model_id)
        result = openfold3_device_fold(
            self.model, host, sampling_steps=int(self.sampling_steps),
            diffusion_samples=int(self.diffusion_samples), distogram=False)

        confs = result.confidence
        best = max(range(len(confs)), key=lambda k: confs[k]["ranking_score"])
        conf = confs[best]
        n_token = int(host.features["restype"].shape[0])
        plddt = token_plddt(conf["plddt_atom"], host.aux["atom_to_token_index"], n_token)
        asym = np.asarray(host.features["asym_id"]).reshape(-1)[:n_token]
        order = {a: i for i, a in enumerate(dict.fromkeys(asym.tolist()))}
        labels = [chains[order[a]] if order[a] < len(chains) else str(a) for a in asym.tolist()]

        pair_iptm = conf.get("pair_chains_iptm")
        summary, chains_in_order = summary_confidences(
            plddt=plddt, pae=np.asarray(conf["pae"], dtype=np.float64), chains=labels,
            ptm=float(conf["ptm"]), iptm=conf.get("iptm"),
            chain_pair_iptm=(_pair_matrix(pair_iptm, len(chains)) if pair_iptm else None),
            chain_ptm=([conf["chain_ptm"][i] for i in range(len(chains))]
                       if conf.get("chain_ptm") else None),
            ranking_score=conf.get("ranking_score"),
            fraction_disordered=conf.get("disorder"), has_clash=conf.get("has_clash"))
        metrics = extract_confidence_metrics(summary, job.chain_of_interest, chains_in_order)
        if job.out_dir is not None:
            self.write_refold(job, metrics, host.features["atom_array"],
                              result.samples[best], conf["plddt_atom"])
        return metrics

    def write_refold(self, job: OracleRequest, metrics: dict, atom_array, coords,
                     plddt_atom) -> tuple[Path, Path]:
        """Write the refold cache upstream's evaluate reads instead of folding again.

        `OracleMetric.compute` folds a design only when its confidence JSON is missing from the
        design's output directory (`oracles/oracle_metric.py:471-520`), so these two files are the
        whole seam between a Tenstorrent refold and every oracle-bound metric column. The JSON is
        written last: a run interrupted mid-fold leaves no cache entry a resumed run would trust.
        """
        from tt_bio.worker import _write_atom_array_structure

        out_dir = Path(job.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{job.example_id}_{self.oracle_name}"
        refolded = out_dir / f"{stem}_refolded.cif"
        _write_atom_array_structure(atom_array, coords, refolded, "cif",
                                    b_factors=plddt_atom * 100.0)
        confidences = out_dir / f"{stem}_confidences.json"
        payload = dict(metrics)
        payload["chain_of_interest"] = job.chain_of_interest
        tmp = confidences.with_suffix(".json.part")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        tmp.rename(confidences)
        return confidences, refolded

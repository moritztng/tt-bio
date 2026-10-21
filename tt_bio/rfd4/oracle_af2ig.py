"""AF2-IG behind the beam-search oracle seam, on the card or on the host.

Upstream's `configs/metrics/refolding_oracles/af2.yaml` scores candidates with AlphaFold2
(`model_1_ptm` for a monomer, `model_1_multimer_v3` for an interface). tt-bio's AF2-IG is the
initial-guess variant of that model: it re-predicts a designed complex starting from the design's
own coordinates, which is the question a binder reward asks, and it needs no MSA, no diffusion and
no seed. It is a near relative of upstream's oracle, not the same configuration, so what a
beam scored by it is worth is `rfd4-quality`'s to grade.

It is also the cheapest oracle tt-bio has for a search loop: `recycles + 1` trunk passes, nothing
else, and `af2ig.fold` takes an already-built model so the chip is opened once for the whole run.

The same adapter serves both arms. `af2ig.fold` only needs a callable `(feats, prev) -> out`, and
`AF2DeviceModel` is a subclass of the torch `AF2Model`, so `backend="tt"` and
`backend="reference"` differ by one constructor and nothing else. That is what lets
`rfd4-quality` grade a beam on the host while the device path is still being built, and it makes
the two arms comparable by construction rather than by assertion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tt_bio.rfd4.oracle import OracleRequest, extract_confidence_metrics, prefix, summary_confidences

logger = logging.getLogger(__name__)

#: One-letter codes by three-letter residue name, for reading a designed chain's sequence back
#: out of the candidate structure. AF2-IG is protein-only, so anything else is a refusal.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def residue_view(atom_array) -> tuple[np.ndarray, np.ndarray]:
    """First-atom index of each residue, and that residue's chain label, in token order."""
    chain_id = np.asarray(atom_array.chain_id)
    res_id = np.asarray(atom_array.res_id)
    changed = np.ones(len(res_id), dtype=bool)
    changed[1:] = (res_id[1:] != res_id[:-1]) | (chain_id[1:] != chain_id[:-1])
    first = np.flatnonzero(changed)
    return first, chain_id[first]


def chain_sequence(structure_text: str, chain: str) -> str:
    """The one-letter sequence of one chain of a candidate, in file order.

    RFD4 emits sequence and structure together, and AF2-IG asks for the designed chain's
    sequence separately from the complex it sits in, so the adapter reads it back off the
    structure rather than carrying a second copy that could disagree with it.
    """
    import io

    import biotite.structure.io.pdb as pdb

    from tt_bio.af2ig import to_pdb_text

    atoms = pdb.PDBFile.read(io.StringIO(to_pdb_text(structure_text))).get_structure(model=1)
    atoms = atoms[np.asarray(atoms.chain_id) == chain]
    if not len(atoms):
        raise ValueError(f"chain {chain!r} is not in the candidate structure")
    first, _ = residue_view(atoms)
    names = np.asarray(atoms.res_name)[first]
    unknown = sorted({n for n in names.tolist() if n not in _THREE_TO_ONE})
    if unknown:
        raise ValueError(f"chain {chain!r} carries non-protein residues {unknown}; "
                         "AF2-IG scores protein complexes only")
    return "".join(_THREE_TO_ONE[n] for n in names.tolist())


def _pair_matrix(pair_chains_iptm: dict, k: int) -> np.ndarray | None:
    """`{i: {j: v}}` keyed by chain position, as the k x k array the summary dict takes."""
    if not pair_chains_iptm or len(pair_chains_iptm) != k:
        return None
    return np.array([[float(pair_chains_iptm[i][j]) for j in range(k)] for i in range(k)])


@dataclass
class AF2IGOracle:
    """Scores a candidate complex with AF2-IG and returns upstream's reward keys.

    `name` is the config key the reward weights read (`oracle`, `oracle_ppi`, ...). `params` is
    the AlphaFold2 npz. `backend` picks the card or the host reference; the reference arm is for
    grading, not for the gated number, which the charter requires on Tenstorrent.
    """

    params: str | Path
    name: str = "oracle"
    backend: str = "tt"
    recycles: int = 3
    target_chain: str | None = None     # None: the one chain that is not the designed chain
    model: object = field(default=None, init=False, repr=False)

    def setup(self) -> None:
        if self.backend not in ("tt", "reference"):
            raise ValueError(f"backend must be 'tt' or 'reference'; got {self.backend!r}")
        if self.model is not None:
            return
        from tt_bio.af2_weights import load_af2_state_dict

        state_dict = load_af2_state_dict(str(self.params))
        if self.backend == "tt":
            from tt_bio.af2 import load_af2_device_model

            # template=True is the binder protocol: the template IS the submitted complex,
            # which is also the initial guess.
            self.model = load_af2_device_model(state_dict, template=True).eval()
        else:
            from tt_bio.af2_reference import load_af2_model

            self.model = load_af2_model(state_dict, template=True).eval()

    def teardown(self) -> None:
        # The weights go; the device stays open, because closing and reopening a chip per
        # consumer is what deadlocks the UMD device-init path (tt_bio/worker.py:861).
        self.model = None

    def fold_batch(self, jobs: list[OracleRequest]) -> list[dict]:
        if self.model is None:
            raise RuntimeError("call setup() before fold_batch()")
        out = []
        for job in jobs:
            try:
                out.append(prefix(self.name, self.score_one(job)))
            except Exception as exc:                        # one bad candidate, not a bad run
                logger.warning("af2ig oracle failed on %s: %s", job.example_id, exc)
                out.append({})
        return out

    def score_one(self, job: OracleRequest) -> dict:
        """One candidate's reward keys. The designed chain is the chain of interest."""
        from tt_bio.af2ig import AF2IGInput, fold

        if job.chain_of_interest is None:
            raise ValueError("AF2-IG scores an interface: name the designed chain as "
                             "the request's chain_of_interest")
        spec = AF2IGInput(
            structure=job.structure,
            binder_sequence=chain_sequence(job.structure, job.chain_of_interest),
            target_chain=self._target_chain(job),
            binder_chain=job.chain_of_interest,
        )
        pred = fold(self.model, spec, recycles=self.recycles)

        first, chains = residue_view(pred.atom_array)
        plddt = np.asarray(pred.b_factors, dtype=float)[first] / 100.0   # AF3 scale, 0..1
        if len(plddt) != pred.tokens:
            raise RuntimeError(f"{len(plddt)} residues for {pred.tokens} tokens")
        pair = _pair_matrix(pred.metrics.get("pair_chains_iptm", {}), 2)
        summary, order = summary_confidences(
            plddt=plddt,
            pae=pred.pae,
            chains=chains,
            ptm=pred.metrics["ptm"],
            iptm=pred.metrics["iptm"],
            chain_pair_iptm=pair,
            chain_ptm=None if pair is None else np.diag(pair),
        )
        return extract_confidence_metrics(summary, job.chain_of_interest, order)

    def _target_chain(self, job: OracleRequest) -> str:
        if self.target_chain is not None:
            return self.target_chain
        import io

        import biotite.structure.io.pdb as pdb

        from tt_bio.af2ig import to_pdb_text

        atoms = pdb.PDBFile.read(io.StringIO(to_pdb_text(job.structure))).get_structure(model=1)
        labels = list(dict.fromkeys(np.asarray(atoms.chain_id).tolist()))
        others = [c for c in labels if c != job.chain_of_interest]
        if len(others) != 1:
            raise ValueError(
                f"AF2-IG takes exactly one target chain beside the designed one; this candidate "
                f"has chains {labels} with {job.chain_of_interest!r} designed. Name the target "
                f"explicitly, or score a multi-chain complex with a co-folding oracle.")
        return others[0]

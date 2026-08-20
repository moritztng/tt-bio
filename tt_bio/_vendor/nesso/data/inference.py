"""Inference datamodule, dataset, and collation utilities.

Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend.
"""

import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
try:  # tt-bio patch: only NessoInferenceDataModule needs Lightning, and the
    # tt-bio runtime does not ship it. InferenceDataset is what we use.
    from lightning.pytorch import LightningDataModule
except ModuleNotFoundError:  # pragma: no cover
    LightningDataModule = object
from numpy.random import RandomState
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tt_bio._vendor.nesso.data import const
from tt_bio._vendor.nesso.data.featurizer import NessoFeaturizer

from tt_bio._vendor.nesso.data.pad import pad_to_max
from tt_bio._vendor.nesso.data.tokenize import tokenize_structure
from tt_bio._vendor.nesso.data.types import Interface, Manifest, Structure, Tokenized

####################################################################################################
# STANDARD AMINO ACIDS
####################################################################################################

STANDARD_AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]


def load_standard_aa_mols(ccd_pkl: Path | None = None) -> dict[str, Any]:
    """Load standard amino acid RDKit mols from a CCD pickle.

    Required for protein sequence featurization. The CCD pickle should contain
    at minimum the 20 standard amino acids + UNK.
    """
    if ccd_pkl is None:
        return {aa: None for aa in STANDARD_AA}
    with ccd_pkl.open("rb") as f:
        ccd_dict = pickle.load(f)  # noqa: S301
    return {aa: ccd_dict.get(aa) for aa in STANDARD_AA}


####################################################################################################
# COLLATE UTILITY
####################################################################################################


def collate(
    data: list[dict[str, Tensor]],
    excluded_keys: set[str] = frozenset({"record", "assay_id"}),
) -> dict[str, Tensor]:
    """Collate a list of feature dicts into a batched dict."""
    if not data:
        return {}

    keys: set[str] = set()
    for d in data:
        keys.update(d.keys())

    collated: dict[str, Any] = {}
    for key in keys:
        sample_val = next((d[key] for d in data if key in d), None)
        values = [
            d[key]
            if key in d
            else (
                torch.zeros_like(sample_val)
                if isinstance(sample_val, torch.Tensor)
                else None
            )
            for d in data
        ]

        if key not in excluded_keys and values and isinstance(values[0], torch.Tensor):
            first_shape = values[0].shape
            if any(v.shape != first_shape for v in values[1:]):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        collated[key] = values

    return collated


INFERENCE_EXCLUDED_KEYS = frozenset({"record", "exception", "record_id"})


def inference_collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    # Must be module-level (picklable) for DataLoader multiprocessing on macOS/Windows.
    return collate(data, excluded_keys=INFERENCE_EXCLUDED_KEYS)


####################################################################################################
# DATASET
####################################################################################################


class InferenceDataset(Dataset):
    """One manifest record per index."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        featurizer: NessoFeaturizer,
        ligand_dir: Path,
        ccd_pkl: Path | None = None,
        use_esm_all_layers: bool = False,
        num_dist_bins: int = 64,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        atoms_per_window_queries: int = 32,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.target_dir = Path(target_dir)
        self.ligand_dir = Path(ligand_dir)
        self.use_esm_all_layers = use_esm_all_layers
        self.num_dist_bins = num_dist_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.atoms_per_window_queries = atoms_per_window_queries
        self._standard_aa_mols = load_standard_aa_mols(
            Path(ccd_pkl) if ccd_pkl is not None else None
        )
        self.featurizer = featurizer

    def _setup_molecules(self, structure: Structure, record_id: str) -> dict[str, Any]:
        """Build residue-name → RDKit Mol mapping for featurization."""
        molecules = {k: v for k, v in self._standard_aa_mols.items() if v is not None}
        lig = const.chain_type_ids["NONPOLYMER"]
        for c in structure.chains:
            if int(c["mol_type"]) != lig:
                continue
            name = str(c["name"]).strip()
            for stem in (f"{record_id}__{name}", name, record_id):
                path = self.ligand_dir / f"{stem}.pkl"
                if path.exists():
                    break
            else:
                raise ValueError(
                    f"No pkl for ligand chain {name!r} in {self.ligand_dir}"
                )
            with path.open("rb") as f:
                mol = pickle.load(f)  # noqa: S301
            for ri in range(int(c["res_idx"]), int(c["res_idx"]) + int(c["res_num"])):
                molecules.setdefault(str(structure.residues[ri]["name"]).strip(), mol)
        return molecules

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.manifest.records[idx]
        record_id = record.id

        try:
            structure = Structure.load(
                self.target_dir / "structures" / f"{record_id}.npz"
            )
            chain_mask = structure.mask.copy()
            structure = structure.remove_invalid_chains(chain_mask)

            ligand_type = const.chain_type_ids["NONPOLYMER"]
            if (
                len(structure.chains) == 2
                and len(structure.interfaces) == 0
                and int(structure.chains[-1]["mol_type"]) == ligand_type
            ):
                new_interfaces = np.array([(0, 1)], dtype=Interface)
                structure = replace(structure, interfaces=new_interfaces)

            token_data, token_bonds = tokenize_structure(structure)
            tokenized = Tokenized(
                tokens=token_data,
                bonds=token_bonds,
                structure=structure,
                record=record,
            )
            random = RandomState(idx)
            molecules = self._setup_molecules(structure, str(record_id))
            features = self.featurizer.process(
                tokenized,
                record=record,
                molecules=molecules,
                random=random,
                num_dist_bins=self.num_dist_bins,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                use_esm_all_layers=self.use_esm_all_layers,
                atoms_per_window_queries=self.atoms_per_window_queries,
                binder_pocket_conditioned_prop=0.0,
                max_tokens=None,
            )
            features["record"] = record
            if record.binder_asym_id is not None:
                features["affinity_token_mask"] = (
                    features["asym_id"] == record.binder_asym_id
                ).float()
            else:
                features["affinity_token_mask"] = (
                    features["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                ).float()
            return features

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error loading record {record_id}: {e}")
            return {"exception": True, "record_id": record_id}

    def __len__(self) -> int:
        return len(self.manifest.records)


####################################################################################################
# DATAMODULE
####################################################################################################


class NessoInferenceDataModule(LightningDataModule):
    """Predict-only datamodule; always batch_size=1."""

    def __init__(
        self,
        target_dir: Path | str,
        esm_emb_dir: Path | str,
        ligand_dir: Path | str,
        manifest_path: Path | str | None = None,
        manifest: Manifest | None = None,
        ccd_pkl: Path | str | None = None,
        num_workers: int = 0,
        use_esm_all_layers: bool = False,
        num_dist_bins: int = 64,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        atoms_per_window_queries: int = 32,
        esm_emb_dim: int = 1280,
        esm_num_layers: int = 33,
    ) -> None:
        super().__init__()
        if manifest is not None:
            self.manifest = manifest
        elif manifest_path is not None:
            self.manifest = Manifest.load(Path(manifest_path))
        else:
            raise ValueError("Must provide either manifest or manifest_path")

        self.target_dir = Path(target_dir)
        self.ligand_dir = Path(ligand_dir)
        self.ccd_pkl = Path(ccd_pkl) if ccd_pkl is not None else None
        self.num_workers = num_workers
        self.use_esm_all_layers = use_esm_all_layers
        self.esm_emb_dim = esm_emb_dim
        self.esm_num_layers = esm_num_layers
        self.num_dist_bins = num_dist_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.atoms_per_window_queries = atoms_per_window_queries
        self.featurizer = NessoFeaturizer(
            esm_emb_dir=Path(esm_emb_dir),
            esm_emb_dim=esm_emb_dim,
            esm_num_layers=esm_num_layers,
        )

    def predict_dataloader(self) -> DataLoader:
        dataset = InferenceDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            ligand_dir=self.ligand_dir,
            ccd_pkl=self.ccd_pkl,
            use_esm_all_layers=self.use_esm_all_layers,
            num_dist_bins=self.num_dist_bins,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            atoms_per_window_queries=self.atoms_per_window_queries,
            featurizer=self.featurizer,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=inference_collate,
        )

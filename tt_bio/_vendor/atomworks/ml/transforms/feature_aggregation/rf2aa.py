"""Transforms to handle final feature aggregation prior to data loading"""

from typing import ClassVar

import numpy as np
import torch
from biotite.structure import AtomArray
from einops import rearrange, repeat

from tt_bio._vendor.atomworks.ml.encoding_definitions import TokenEncoding
from tt_bio._vendor.atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from tt_bio._vendor.atomworks.ml.transforms.atom_array import AddProteinTerminiAnnotation
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.transforms.bonds import AddRF2AABondFeaturesMatrix, AddRF2AATraversalDistanceMatrix
from tt_bio._vendor.atomworks.ml.transforms.msa.msa import FeaturizeMSALikeRF2AA
from tt_bio._vendor.atomworks.ml.transforms.template import RF2AATemplate
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts


class AggregateFeaturesLikeRF2AA(Transform):
    """
    Combines features into the correct shapes for RF2AA.

    Initialization arguments:
        encoding (TokenEncoding): The TokenEncoding to use (we only care about `encoding.n_token` when aggregating features)
        use_negative_interface_examples (bool): Whether to use negative interface examples when training. Currently, RF2AA only uses negative examples during fine-tuning.
        unclamp_loss_probability (float): The probability of unclamping the loss (normally, we apply a "leaky" clamp to avoid zero gradients). Like AF-2, we periodically (10% of the time) assess the FAPE against the unclamped loss.

    Inputs for RFAA (from the RF2AA Supplementary Information, modified for accuracy with current model architecture and data loading):
        ```
        Remaining features such as MSAs and templates are handled identically for proteins to RF2.
        The coordinate dimension, 36, reflects the maximum amount of heavy atoms and hydrogens possible in a residue or base.
        The small molecule tokens are appended to the first sequence in all the MSA features and the remaining MSA sequences are initialized with gap tokens.
        Small molecules receive empty template features which are concatenated block diagonally to the protein features.

        | Input (dimension)         | Description                                                                                                                                        |
        |---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
        | msa_masked                | (Nnum_clusters, L, 164) Clustered MSA with some portions of the sequences masked. For atom nodes, the first sequence has its respective atom       |
                                      tokens and then remaining sequences are filled with gap tokens. (80 raw msa, 80 cluster statistics, 2 insertions/deletions, 2 Nterm/Cterm)         |
        | msa_full                  | (Nnum_sequences, L, 83) Full MSA clipped at 1024 sequences (80). Also includes raw insertion counts (1), and N/C-terminal information (2).         |
        | seq                       | (L, 80) First row of the MSA. In this case, the protein sequence and any atom tokens, including mask tokens.                                       |
        | idx                       | (L) Residue index of each residue in the input. This input must be provided for atom nodes but has no semantic                                     |
                                      meaning (it is unused by the network).                                                                                                             |
        | bond_feats                | (L, L, 7) Pairwise bond adjacency matrix. Pairs of residues are either single, double, triple, aromatic, residue-residue, residue-atom or other.   |
        | dist_matrix               | (L, L) Minimum amount of bonds to traverse between two nodes. This is 0 between all protein nodes.                                                 |
        | chirals                   | (Lnum_chiral_centers, 5) All orderings of 4 atoms around a chiral center (first four dimensions) and the ideal pseudo-dihedral angle               |
                                      formed by that ordering of atoms (fifth dimension).                                                                                                |
        | atom_frames               | (Lnum_atoms, 3, 2) Indices that form frames for each atom node in the input. The second dimension represents that there are three                  |
                                      atoms in each frame.                                                                                                                               |
                                      The third dimension represents an offset in the node dimension because atom frames go across nodes and the absolute index in the atom dimension.   |
        | t1d                       | (Nnum_templates, L, 80) 1D template feature. First, 79 represent the "sequence" (residue/atom types) of the templated structure.                   |
                                      Last dimension represents residue wise template confidence.                                                                                        |
        | t2d                       | (Nnum_templates, L, L, 64) 2D template information which gives the binned distances and angles between frames (N-C_alpha-C for proteins,           |
                                      designated atom frame for atoms) # NOTE: This is generated by RF2AA itself.                                                                                                                  |
        | alpha_t                   | (Nnum_templates, L, 30) Sidechain torsion angles from templates (10 angles x sin, cos and whether the angle exists in the                          |
                                      structure for each residue)                                                                                                                        |
        | msa_prev                  | (Nnum_clusters, L, Cm) Recycled MSA features. Cm=256 (number of 1D channels)                                                                       |
        | pair_prev                 | (L, L, Cp) Recycled pair features. Cp=192 (number of 2D channels)                                                                                  |
        | state_prev                | (L, Cs) Recycled state features. Cs=32 (number of 3D channels)                                                                                     |
        | xyz_prev                  | (L, 36, 3) Recycled XYZ coordinates. On first iteration, this is set to the coordinates from the first template.                                   |
                                      If no templates, coordinates are initialized at the origin with random noise (between -2.5 and 2.5Å) applied.                                      |
        | sc_torsions_prev          | (L, 30) Recycled predicted sidechain torsion angles.                                                                                               |
        ```
    """

    # In theory, all of them, but we only list the top-level transforms since the rest are handled by the dependency tree
    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        FeaturizeMSALikeRF2AA,
        AddProteinTerminiAnnotation,
        AddRF2AABondFeaturesMatrix,
        AddRF2AATraversalDistanceMatrix,
        "AddRFTemplates",
        "FeaturizeTemplatesLikeRF2AA",
        "AddRF2AAChiralFeatures",
        "CreateSymmetryCopyAxisLikeRF2AA",
        "SortLikeRF2AA",
        "AddPostCropMoleculeEntityToFreeFloatingLigands",
    ]

    def __init__(
        self,
        encoding: TokenEncoding,
        use_negative_interface_examples: bool,
        unclamp_loss_probability: float,
        black_hole_init: bool = True,
        black_hole_init_coords: np.ndarray = RF2AATemplate.RF2AA_INIT_TEMPLATE_COORDINATES,
        black_hole_init_noise_scale: float = 5.0,  # Å
    ):
        """
        Aggregates all features from the processed `data` necessary to run the RF2AA model.

        Args:
            - encoding (TokenEncoding): The encoding scheme used for tokenizing the input sequences and
                coordinates.
            - use_negative_interface_examples (bool): Flag indicating whether to use negative
                interface examples.
            - unclamp_loss_probability (float): Probability of unclamping the loss during training.
                Must be between 0 and 1 (inclusive).
            - black_hole_init (bool, optional): Flag to enable black hole initialization.
                Black hole initialization means that the `xyz_prev` is set to the default
                `black_hole_init_coords` (default: ChemData().INIT_CRDS) with some noise
                added. Defaults to True. In the case where this is false, `xyz_prev` is initialized
                to the coordinates of the first template.
            - black_hole_init_coords (np.ndarray, optional): Initial coordinates for black hole
                initialization. Defaults to ChemData().INIT_CRDS. Only used if `black_hole_init`
                is True.
            - black_hole_init_noise_scale (float, optional): Scale of noise to apply during black
              hole initialization. Defaults to 5.0 Å. Only used if `black_hole_init` is True.
              The noise is drawn uniformly from the range [-`black_hole_init_noise_scale`/2,
              `black_hole_init_noise_scale`/2].
        """
        self.encoding = encoding
        self.use_negative_interface_examples = use_negative_interface_examples
        self.unclamp_loss_probability = unclamp_loss_probability
        self.black_hole_init = black_hole_init
        self.black_hole_init_coords = black_hole_init_coords
        self.black_hole_init_noise_scale = black_hole_init_noise_scale

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["features_per_recycle_dict", "template_feat", "encoded"])  # TODO: Add other keys
        check_is_instance(data, "atom_array", AtomArray)  # TODO: Add other checks
        check_atom_array_annotation(data, ["molecule_entity", "molecule_iid", "is_N_terminus", "is_C_terminus"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        data_outputs = {}

        # +------------------------------------------------------------------------+
        # +------------------------- MSA-related features -------------------------+
        # +------------------------------------------------------------------------+

        features_per_recycle_dict = data["features_per_recycle_dict"]
        # Sequence
        data_outputs["seq"] = torch.stack(
            features_per_recycle_dict["first_row_of_msa"], dim=0
        )  # [n_recycles, n_tokens_across_chains] (int)

        # Main MSA features (direct)
        cluster_representatives_msa_masked = torch.stack(
            features_per_recycle_dict["cluster_representatives_msa_masked"], dim=0
        )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains] (int)
        # cluster_representatives_has_insertion = torch.stack(features_per_recycle_dict["cluster_representatives_has_insertion"], dim=0) # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains] (bool) [UNUSED IN RF2AA]
        cluster_representatives_insertion_value = torch.stack(
            features_per_recycle_dict["cluster_representatives_insertion_value"], dim=0
        )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains] (float)

        # Main MSA features (cluster profiles)
        cluster_insertion_mean = torch.stack(
            features_per_recycle_dict["cluster_insertion_mean"], dim=0
        )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains] (float)
        cluster_profile = torch.stack(
            features_per_recycle_dict["cluster_profile"], dim=0
        )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)

        # Extra MSA
        extra_msa = torch.stack(
            features_per_recycle_dict["extra_msa"], dim=0
        )  # [n_recycles, n_extra_msa, n_tokens_across_chains] (int)
        # extra_msa_has_insertion = torch.stack(features_per_recycle_dict["extra_msa_has_insertion"], dim=0) # [n_recycles, n_extra_msa, n_tokens_across_chains] (bool) [UNUSED IN RF2AA]
        extra_msa_insertion_value = torch.stack(
            features_per_recycle_dict["extra_msa_insertion_value"], dim=0
        )  # [n_recycles, n_extra_msa, n_tokens_across_chains] (float)

        # Step 2: Gather N-terminal and C-terminal information from the atom array
        # Assert that the chain count is the same as the number of N- and C-termini
        token_starts = get_token_starts(atom_array)
        token_wise_atom_array = atom_array[token_starts]
        token_is_n_terminus = torch.from_numpy(
            token_wise_atom_array.is_N_terminus.astype(np.int64)
        )  # [n_tokens_across_chains] (int)
        token_is_c_terminus = torch.from_numpy(
            token_wise_atom_array.is_C_terminus.astype(np.int64)
        )  # [n_tokens_across_chains] (int)

        # Step 3: Concatenate all the features for the cluster representative and extra MSA into the correct dimensions
        n_recycles = cluster_representatives_msa_masked.shape[0]
        n_msa_cluster_representatives = cluster_representatives_msa_masked.shape[1]
        n_extra_msa = extra_msa.shape[1]

        data_outputs["msa_masked"] = torch.concatenate(
            [
                torch.nn.functional.one_hot(
                    cluster_representatives_msa_masked, num_classes=self.encoding.n_tokens
                ),  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (bool)
                cluster_profile,  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)
                rearrange(cluster_representatives_insertion_value, "... -> ... 1"),  # [..., 1] (float)
                rearrange(cluster_insertion_mean, "... -> ... 1"),  # [..., 1] (float)
                repeat(
                    token_is_n_terminus, "l -> r s l 1", r=n_recycles, s=n_msa_cluster_representatives
                ),  # [..., 1] (int) # NOTE: Repeats the N-terminus token across all cluster representatives; may not be desired behavior
                repeat(
                    token_is_c_terminus, "l -> r s l 1", r=n_recycles, s=n_msa_cluster_representatives
                ),  # [..., 1] (int) # NOTE: Repeats the C-terminus token across all cluster representatives; may not be desired behavior
            ],
            dim=-1,  # Concatenate along the last dimension
        )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains, 2 * n_tokens + 4] (float)

        data_outputs["msa_full"] = torch.concatenate(
            [
                torch.nn.functional.one_hot(
                    extra_msa,
                    num_classes=self.encoding.n_tokens,
                ),  # [n_recycles, n_extra_msa, n_tokens_across_chains, n_tokens] (float32)
                rearrange(extra_msa_insertion_value, "... -> ... 1"),  # [..., 1] (float)
                repeat(
                    token_is_n_terminus, "l -> r s l 1", r=n_recycles, s=n_extra_msa
                ),  # [..., 1] (int) # NOTE: Repeats the N-terminus token across all extra MSA; may not be desired behavior
                repeat(
                    token_is_c_terminus, "l -> r s l 1", r=n_recycles, s=n_extra_msa
                ),  # [..., 1] (int) # NOTE: Repeats the C-terminus token across all extra MSA; may not be desired behavior
            ],
            dim=-1,  # Concatenate along the last dimension
        )  # [n_recycles, n_extra_msa, n_tokens_across_chains, n_tokens + 3] (float)

        # +------------------------------------------------------------------------+
        # +------------------------ Bond-related features -------------------------+
        # +------------------------------------------------------------------------+

        data_outputs["bond_feats"] = data[
            "rf2aa_bond_features_matrix"
        ].long()  # [n_tokens_across_chains, n_tokens_across_chains] (int)
        data_outputs["dist_matrix"] = data[
            "rf2aa_traversal_distance_matrix"
        ]  # [n_tokens_across_chains, n_tokens_across_chains] (int)

        # +------------------------------------------------------------------------+
        # +----------------------------- Atom frames ------------------------------+
        # +------------------------------------------------------------------------+

        rf2aa_atom_frames = data["rf2aa_atom_frames"]  # [n_tokens_across_chains, 3, 2] (int)

        # Index to only the atomized tokens
        atomized_tokens = token_wise_atom_array.atomize

        if np.any(atomized_tokens):
            rf2aa_atom_frames = rf2aa_atom_frames[atomized_tokens]  # [n_atomized_tokens, 3, 2] (int)
        else:
            # If there are no atomized tokens, we need to add a dummy atom frame
            rf2aa_atom_frames = torch.zeros((0, 3, 2), dtype=torch.int64)

        data_outputs["atom_frames"] = rf2aa_atom_frames  # [n_atomized_tokens, 3, 2] (int)

        # +------------------------------------------------------------------------+
        # +-------------------------- Residue Indices -----------------------------+
        # +------------------------------------------------------------------------+

        # NOTE: `idx` values for non-polymers hold no semantic value and are unused by the network
        # ...get the delta between consecutive residue indices
        delta = token_wise_atom_array.res_id[1:] - token_wise_atom_array.res_id[:-1]

        # ...ensure that the delta is non-negative
        delta[delta <= 0] = 0

        # ...between chain instances, add 100 to the delta
        delta += 100 * (token_wise_atom_array.chain_iid[1:] != token_wise_atom_array.chain_iid[:-1])

        # ...add the first residue index to the delta to get the absolute residue index (with 100 added between chain instances)
        data_outputs["idx_pdb"] = torch.from_numpy(
            np.cumsum(np.concatenate([token_wise_atom_array.res_id[0:1], delta], axis=0), axis=0).astype(np.int64)
        )

        # +------------------------------------------------------------------------+
        # +------------------------------ Chirals ---------------------------------+
        # +------------------------------------------------------------------------+

        # chirals , e.g. (60, 5)
        data_outputs["chirals"] = data["chiral_feats"]  # [n_chirals, 5]

        # +------------------------------------------------------------------------+
        # +----------------------------- Templates --------------------------------+
        # +------------------------------------------------------------------------+

        # xyz_t & mask_t , e.g. (1, 113, 36, 3)
        xyz_t = data["template_feat"]["xyz"]  # [n_templates, n_token, n_atoms_per_token, 3]
        mask_t = data["template_feat"]["mask"]  # [n_templates, n_token]
        # t1d (1, 113, 80)
        t1d = data["template_feat"]["t1d"]  # [n_templates, n_token, n_tokens(80)]

        data_outputs["xyz_t"] = xyz_t
        data_outputs["mask_t"] = mask_t
        data_outputs["t1d"] = t1d

        # Initialize the 3D track: For `black_hole_init` = True, we initialize the 3D track around the
        #  origin with some noise to break the symmetry. Otherwise, we initialize the 3D track to the
        #  coordinates of the first template.
        # xyz_prev , e.g. (1, 113, 36, 3)
        n_token = xyz_t.shape[1]
        if self.black_hole_init:
            # ... initialize `xyz_prev` around the origin with some noise to break the symmetry
            xyz_prev = self.black_hole_init_coords.reshape(1, self.encoding.n_atoms_per_token, 3).repeat(n_token, 1, 1)
            xyz_prev += (
                torch.rand(n_token, 1, 3) * self.black_hole_init_noise_scale - self.black_hole_init_noise_scale / 2
            )
            mask_prev = torch.zeros((n_token, self.encoding.n_atoms_per_token), dtype=bool)
        else:
            # ... initialize `xyz_prev` to the coordinates of the first template
            xyz_prev = xyz_t[0].clone()
            mask_prev = mask_t[0].clone()

        data_outputs["xyz_prev"] = xyz_prev
        data_outputs["mask_prev"] = mask_prev

        # +------------------------------------------------------------------------+
        # +----- Inputs for loss computation (not input features to the model) ----+
        # +------------------------------------------------------------------------+
        # True coordinates (ground truth)
        data_outputs["xyz"] = data["encoded"]["xyz"]  # [n_symmetry, n_tokens, n_atoms_per_token, 3]
        # Mask coordinates (ground truth)
        data_outputs["mask"] = data["encoded"]["mask"]  # [n_symmetry, n_tokens, n_atoms_per_token]
        # Sequence (ground truth) -- not used in the model but useful for debugging
        data_outputs["seq_gt"] = data["encoded"]["seq"]  # [n_tokens]

        # Note where we applied the BERT mask so we know where to compute the masked token recovery loss
        data_outputs["mask_msa"] = torch.stack(
            features_per_recycle_dict["bert_mask_position"], dim=0
        )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains] (bool)

        # Original MSA, to be used as ground truth when computing a masked token recovery loss
        data_outputs["msa"] = torch.stack(
            features_per_recycle_dict["cluster_representatives_msa_ground_truth"], dim=0
        )  # [n_recycles, n_msa_cluster_representatives, n_tokens_across_chains] (int)

        # Molecule entity, used during loss computation to determine possible symmetric ligand swaps
        if "post_crop_molecule_entity" in token_wise_atom_array.get_annotation_categories():
            ch_label = token_wise_atom_array.post_crop_molecule_entity.astype(np.int64)
        else:
            ch_label = token_wise_atom_array.molecule_entity.astype(np.int64)
        data_outputs["ch_label"] = torch.from_numpy(ch_label)  # [n_tokens_across_chains] (int)

        # Matrix that notes whether two token indices have the same molecule iid & entity,
        #  used for logging inter-molecule losses during validation.
        _make_is_same_matrix = lambda x: x.unsqueeze(0) == x.unsqueeze(0).T  # noqa
        is_same_molecule_iid = _make_is_same_matrix(
            torch.from_numpy(token_wise_atom_array.molecule_iid.astype(np.int64))
        )
        data_outputs["same_chain"] = is_same_molecule_iid
        # NOTE: This is not a feature input to the network; only used for logging inter-molecule losses

        ### PATCH
        # To avoid the `KeyError` in RF2AA symmetry resolution in loss, we ensure that everything
        #  in the same chain comes from the same entity by updating ch_label
        ch_label = data_outputs["ch_label"]
        same_chain = data_outputs["same_chain"]
        idx = 0
        while idx < len(ch_label):
            label = ch_label[idx]
            same_chain_row = same_chain[idx, idx:]
            for col in range(len(same_chain_row)):
                if same_chain_row[col] == 0:
                    block_size = col
                    break
                elif col == len(same_chain_row) - 1:
                    block_size = len(same_chain_row)
            # set the label for the entire chain to the label of the first token
            ch_label[idx : idx + block_size] = label
            idx += block_size
        data_outputs["ch_label"] = ch_label
        ###

        # Periodically (self.unclamp_loss_probability of the time, e.g., 10%), we unclamp the loss to assess FAPE against the unclamped loss
        unclamp = torch.rand(1) < self.unclamp_loss_probability
        data_outputs["unclamp"] = torch.tensor(unclamp)

        # Whether we are performing negative sampling of interfaces; currently, only set to True during fine-tuning
        negative = bool(self.use_negative_interface_examples)
        data_outputs["negative"] = torch.tensor(negative)
        data_outputs["example_id"] = data["example_id"]
        data_outputs["task"] = "sm_compl" if np.any(token_wise_atom_array.atomize) else "poly_only"
        data_outputs["symmgp"] = "C1"  # (Assume no special symmetry group -- C1=cyclical 1 == identity)

        # Map `chain_iid` from integers to strings
        int_to_chain_iid = {v: k for k, v in data["encoded"]["chain_iid_to_int"].items()}
        vectorized_map = np.vectorize(lambda x: int_to_chain_iid[x])
        chain_iid_token_lvl = vectorized_map(data["encoded"]["chain_iid"])

        # Add in additional ground truth information needed for loss computation and evaluation
        # (We may already have ground_truth in the data, i.e., during validation, when we pass extra information for evaluation)
        if "ground_truth" not in data:
            data["ground_truth"] = {}
        data["ground_truth"].update(
            {
                "chain_iid_token_lvl": chain_iid_token_lvl,  # numpy.ndarray of strings with shape (n_tokens,)
            }
        )

        # `nan_to_num` for the xyz_t, xyz_prev, and xyz features
        data_outputs["xyz_t"] = torch.nan_to_num(data_outputs["xyz_t"])
        data_outputs["xyz_prev"] = torch.nan_to_num(data_outputs["xyz_prev"])
        data_outputs["xyz"] = torch.nan_to_num(data_outputs["xyz"])

        data["feats"] = data_outputs
        return data

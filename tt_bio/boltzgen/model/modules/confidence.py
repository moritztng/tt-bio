import torch
from torch import nn
from torch.nn.functional import pad

import tt_bio.boltzgen.model.layers.initialize as init
from tt_bio.data import const
from tt_bio.boltzgen.model.layers.confidence_utils import (
    compute_aggregated_metric,
    compute_ptms,
)

from tt_bio.boltz2 import conf_heads_bins, device_conf_heads
from tt_bio.tenstorrent import PairformerModule as TTPairformerModule
from tt_bio.boltzgen.model.modules.encoders import RelativePositionEncoder
from tt_bio.boltzgen.model.modules.trunk import (
    ContactConditioning,
)
from tt_bio.boltzgen.model.modules.utils import LinearNoBias


class ConfidenceModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        num_dist_bins=64,
        token_level_confidence=True,
        max_dist=22,
        add_s_to_z_prod=False,
        confidence_args: dict = None,
        return_latent_feats=False,
        conditioning_cutoff_min=None,
        conditioning_cutoff_max=None,
        bond_type_feature=False,
        add_z_input_to_z=None,  # old checkpoint compatibility
        add_s_input_to_s=None,  # old checkpoint compatibility
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.no_update_s = pairformer_args.get("no_update_s", False)
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)
        self.token_level_confidence = token_level_confidence

        self.s_to_z = LinearNoBias(token_s, token_z)
        self.s_to_z_transpose = LinearNoBias(token_s, token_z)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
            init.gating_init_(self.s_to_z_prod_out.weight)

        self.s_inputs_norm = nn.LayerNorm(token_s)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.add_s_input_to_s = add_s_input_to_s
        if add_s_input_to_s:
            self.s_input_to_s = LinearNoBias(token_s, token_s)
            init.gating_init_(self.s_input_to_s.weight)

        self.add_z_input_to_z = add_z_input_to_z
        if add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(token_z)
            self.token_bonds = nn.Linear(
                1,
                token_z,
                bias=False,
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z,
                cutoff_min=conditioning_cutoff_min,
                cutoff_max=conditioning_cutoff_max,
            )

        # Direct ttnn — confidence runs its own (smaller, 8-block) pairformer.
        num_heads = pairformer_args.get("num_heads", 16)
        self.pairformer_stack = TTPairformerModule(
            n_blocks=pairformer_args["num_blocks"],
            tri_att_head_dim=pairformer_args.get("pairwise_head_width", 32),
            tri_att_n_heads=pairformer_args.get("pairwise_num_heads", 4),
            att_head_dim=token_s // num_heads,
            att_n_heads=num_heads,
            transform_s=True,
        )
        self.return_latent_feats = return_latent_feats

        self.confidence_heads = ConfidenceHeads(
            token_s,
            token_z,
            token_level_confidence=token_level_confidence,
            **confidence_args,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        x_pred,  # Float['bm m 3']
        feats,
        pred_distogram_logits,
        multiplicity=1,
        s_diffusion=None,
        run_sequentially=False,
        use_kernels: bool = False,
    ):
        s_inputs = self.s_inputs_norm(s_inputs)
        if not self.no_update_s:
            s = self.s_norm(s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)

        # Everything up to here, and the pair terms below, read no sample: the sequential path
        # used to rebuild them once per diffusion sample. Built once, they are the same tensors.
        z = self.z_norm(z)
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        if not (run_sequentially and multiplicity > 1):
            return self._sample(s_inputs, s, z, x_pred, feats, pred_distogram_logits,
                                multiplicity, use_kernels)

        assert z.shape[0] == 1, "Not supported with batch size > 1"
        out_dicts = [
            self._sample(s_inputs, s, z, x_pred[i : i + 1], feats, pred_distogram_logits,
                         1, use_kernels)
            for i in range(multiplicity)
        ]
        out_dict = {}
        for key in out_dicts[0]:
            if key != "pair_chains_iptm" and key != "chain_pair_ipsae":
                out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
            else:
                pair_chains_dict = {}
                for chain_idx1 in out_dicts[0][key]:
                    chains_dict = {}
                    for chain_idx2 in out_dicts[0][key][chain_idx1]:
                        chains_dict[chain_idx2] = torch.cat(
                            [out[key][chain_idx1][chain_idx2] for out in out_dicts],
                            dim=0,
                        )
                    pair_chains_dict[chain_idx1] = chains_dict
                out_dict[key] = pair_chains_dict
        return out_dict

    def _sample(self, s_inputs, s, z, x_pred, feats, pred_distogram_logits, multiplicity,
                use_kernels):
        """The per-sample part: distogram embedding, pairformer, heads."""
        s = s.repeat_interleave(multiplicity, 0)
        z = z.repeat_interleave(multiplicity, 0)
        s_inputs = s_inputs.repeat_interleave(multiplicity, 0)

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)
        z = z + distogram

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        # Boltz-2's device heads: pae/pde/TM are reduced where the pairformer left z, and only
        # the three [1, n, n] maps come back instead of z itself.
        heads_device = device_conf_heads(self, multiplicity, "boltzgen")
        seq_len = z.shape[1]
        s_t, z_t = self.pairformer_stack(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels,
            **({"keep_z_device": True} if heads_device is not None else {}),
        )

        # AF3 has residual connections, we remove them
        s = s_t
        z = z_t

        pair_dev = None
        if heads_device is not None:
            pair_dev = heads_device(z, feats["asym_id"], conf_heads_bins(heads_device, mask),
                                    seq_len, int(z.shape[1]))
            z = None

        out_dict = {}

        if self.return_latent_feats:
            out_dict["s_conf"] = s
            out_dict["z_conf"] = z

        # confidence heads
        out_dict.update(
            self.confidence_heads(
                s=s,
                z=z,
                x_pred=x_pred,
                d=d,
                feats=feats,
                multiplicity=multiplicity,
                pred_distogram_logits=pred_distogram_logits,
                pair_dev=pair_dev,
            )
        )
        return out_dict


class ConfidenceHeads(nn.Module):
    def __init__(
        self,
        token_s,
        token_z,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        token_level_confidence=True,
        use_separate_heads: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.token_level_confidence = token_level_confidence
        self.use_separate_heads = use_separate_heads

        if self.use_separate_heads:
            self.to_pae_intra_logits = LinearNoBias(token_z, num_pae_bins)
            self.to_pae_inter_logits = LinearNoBias(token_z, num_pae_bins)
        else:
            self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

        if self.use_separate_heads:
            self.to_pde_intra_logits = LinearNoBias(token_z, num_pde_bins)
            self.to_pde_inter_logits = LinearNoBias(token_z, num_pde_bins)
        else:
            self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)

        if self.token_level_confidence:
            self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
            self.to_resolved_logits = LinearNoBias(token_s, 2)
        else:
            self.to_plddt_logits = LinearNoBias(
                token_s, num_plddt_bins * self.max_num_atoms_per_token
            )
            self.to_resolved_logits = LinearNoBias(
                token_s, 2 * self.max_num_atoms_per_token
            )

    def _pair_logits(self, z, feats):
        """The host pae and pde logits over ``z``."""
        if not self.use_separate_heads:
            return self.to_pae_logits(z), self.to_pde_logits(z + z.transpose(1, 2))
        asym_id_token = feats["asym_id"]
        is_same_chain = asym_id_token.unsqueeze(-1) == asym_id_token.unsqueeze(-2)
        same = is_same_chain.float().unsqueeze(-1)
        different = (~is_same_chain).float().unsqueeze(-1)
        pae_logits = (self.to_pae_inter_logits(z) * different
                      + self.to_pae_intra_logits(z) * same)
        z_sym = z + z.transpose(1, 2)
        pde_logits = (self.to_pde_inter_logits(z_sym) * different
                      + self.to_pde_intra_logits(z_sym) * same)
        return pae_logits, pde_logits

    def forward(
        self,
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        x_pred,  # Float['bm m 3']
        d,
        feats,
        pred_distogram_logits,
        multiplicity=1,
        pair_dev=None,
    ):
        # `pair_dev` is {"pae", "pde", "tm"} already reduced over the bins by
        # `tenstorrent.ConfidenceHeadsDevice`, and `z` is None: the logits never exist on the host.
        if pair_dev is not None:
            pae_logits = pde_logits = None
        else:
            pae_logits, pde_logits = self._pair_logits(z, feats)
        resolved_logits = self.to_resolved_logits(s)
        plddt_logits = self.to_plddt_logits(s)

        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
        is_protein_token = (token_type == const.chain_type_ids["PROTEIN"]).float()

        if self.token_level_confidence:
            plddt = compute_aggregated_metric(plddt_logits)
            token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
            complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(
                dim=-1
            )

            is_contact = (d < 8).float()
            is_different_chain = (
                feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)
            ).float()
            is_different_chain = is_different_chain.repeat_interleave(multiplicity, 0)
            token_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
                dim=-1,
            ).values
            token_non_interface_mask = (1 - token_interface_mask) * (
                1 - is_ligand_token
            )
            iplddt_weight = (
                is_ligand_token * ligand_weight
                + token_interface_mask * interface_weight
                + token_non_interface_mask * non_interface_weight
            )
            complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(
                dim=-1
            ) / torch.sum(token_pad_mask * iplddt_weight, dim=-1)

        else:
            # token to atom conversion for resolved logits
            B, N, _ = resolved_logits.shape
            resolved_logits = resolved_logits.reshape(
                B, N, self.max_num_atoms_per_token, 2
            )

            arange_max_num_atoms = (
                torch.arange(self.max_num_atoms_per_token)
                .reshape(1, 1, -1)
                .to(resolved_logits.device)
            )
            max_num_atoms_mask = (
                feats["atom_to_token"].sum(1).unsqueeze(-1) > arange_max_num_atoms
            )
            resolved_logits = resolved_logits[:, max_num_atoms_mask.squeeze(0)]
            resolved_logits = pad(
                resolved_logits,
                (
                    0,
                    0,
                    0,
                    int(
                        feats["atom_pad_mask"].shape[1]
                        - feats["atom_pad_mask"].sum().item()
                    ),
                ),
                value=0,
            )
            plddt_logits = plddt_logits.reshape(B, N, self.max_num_atoms_per_token, -1)
            plddt_logits = plddt_logits[:, max_num_atoms_mask.squeeze(0)]
            plddt_logits = pad(
                plddt_logits,
                (
                    0,
                    0,
                    0,
                    int(
                        feats["atom_pad_mask"].shape[1]
                        - feats["atom_pad_mask"].sum().item()
                    ),
                ),
                value=0,
            )
            atom_pad_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)
            plddt = compute_aggregated_metric(plddt_logits)

            complex_plddt = (plddt * atom_pad_mask).sum(dim=-1) / atom_pad_mask.sum(
                dim=-1
            )
            token_type = feats["mol_type"].float()
            atom_to_token = feats["atom_to_token"].float()
            chain_id_token = feats["asym_id"].float()
            atom_type = torch.bmm(atom_to_token, token_type.unsqueeze(-1)).squeeze(-1)
            is_ligand_atom = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
            d_atom = torch.cdist(x_pred, x_pred)
            is_contact = (d_atom < 8).float()
            chain_id_atom = torch.bmm(
                atom_to_token, chain_id_token.unsqueeze(-1)
            ).squeeze(-1)
            is_different_chain = (
                chain_id_atom.unsqueeze(-1) != chain_id_atom.unsqueeze(-2)
            ).float()

            atom_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_atom).unsqueeze(-1),
                dim=-1,
            ).values
            atom_non_interface_mask = (1 - atom_interface_mask) * (1 - is_ligand_atom)
            iplddt_weight = (
                is_ligand_atom * ligand_weight
                + atom_interface_mask * interface_weight
                + atom_non_interface_mask * non_interface_weight
            )

            complex_iplddt = (plddt * feats["atom_pad_mask"] * iplddt_weight).sum(
                dim=-1
            ) / torch.sum(feats["atom_pad_mask"] * iplddt_weight, dim=-1)

        # Compute the gPDE and giPDE
        if pair_dev is not None:
            pde, pae = pair_dev["pde"], pair_dev["pae"]
        else:
            pde = compute_aggregated_metric(pde_logits, end=32)
            pae = compute_aggregated_metric(pae_logits, end=32)
        pred_distogram_prob = nn.functional.softmax(
            pred_distogram_logits, dim=-1
        ).repeat_interleave(multiplicity, 0)
        contacts = torch.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype).to(
            pred_distogram_prob.device
        )
        contacts[:, :, :, :20] = 1.0
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (
                1
                - torch.eye(
                    token_pad_mask.shape[1], device=token_pad_mask.device
                ).unsqueeze(0)
            )
        )
        # added code
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        selected_ligand_mask = is_ligand_token * (asym_id == 1).float()
        protein_pair_mask = token_pad_pair_mask * (
            is_protein_token.unsqueeze(-1) * is_protein_token.unsqueeze(-2)
        )
        protein_pair_mask_weighted = protein_pair_mask * prob_contact
        protein_pae_value_weighted = (pae * protein_pair_mask_weighted).sum(
            dim=(1, 2)
        ) / (protein_pair_mask_weighted.sum(dim=(1, 2)) + 1e-5)
        protein_pae_value_unweighted = (pae * protein_pair_mask).sum(dim=(1, 2)) / (
            protein_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        ligand_pair_mask = token_pad_pair_mask * (
            selected_ligand_mask.unsqueeze(-1) * selected_ligand_mask.unsqueeze(-2)
        )

        ligand_pair_mask_weighted = ligand_pair_mask * prob_contact
        ligand_pae_value_weighted = (pae * ligand_pair_mask_weighted).sum(
            dim=(1, 2)
        ) / (ligand_pair_mask_weighted.sum(dim=(1, 2)) + 1e-5)
        ligand_pae_value_unweighted = (pae * ligand_pair_mask).sum(dim=(1, 2)) / (
            ligand_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        interface_protein_mask = torch.max(
            (d < 8).float()
            * is_protein_token.unsqueeze(-1)
            * selected_ligand_mask.unsqueeze(-2),
            dim=-1,
        ).values
        protein_ligand_interface_pair_mask = (
            token_pad_pair_mask
            * interface_protein_mask.unsqueeze(-1)
            * selected_ligand_mask.unsqueeze(-2)
            * (d < 8).float()
        )
        protein_ligand_interface_pair_mask_weighted = (
            protein_ligand_interface_pair_mask * prob_contact
        )
        interface_pae_value_weighted = (
            pae * protein_ligand_interface_pair_mask_weighted
        ).sum(dim=(1, 2)) / (
            protein_ligand_interface_pair_mask_weighted.sum(dim=(1, 2)) + 1e-5
        )
        interface_pae_value_unweighted = (pae * protein_ligand_interface_pair_mask).sum(
            dim=(1, 2)
        ) / (protein_ligand_interface_pair_mask.sum(dim=(1, 2)) + 1e-5)
        # finish
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(dim=(1, 2)) / token_pair_mask.sum(
            dim=(1, 2)
        )
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (
            asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        )
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / (
            token_interface_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        # Compute design metric PDE
        is_chain_design_token = (
            feats["chain_design_mask"].float()
            if feats["design_mask"].sum() > 0
            else feats["design_mask"].float()
        )
        is_target_token = (
            1 - feats["chain_design_mask"].float()
            if feats["design_mask"].sum() > 0
            else feats["design_mask"].float()
        )
        target_designchain_inter_mask = token_pad_pair_mask * (
            is_chain_design_token[:, :, None] * is_target_token[:, None, :]
            + is_target_token[:, :, None] * is_chain_design_token[:, None, :]
        )
        interaction_pae = (pae * target_designchain_inter_mask).sum(dim=(1, 2)) / (
            target_designchain_inter_mask.sum(dim=(1, 2)) + 1e-5
        )
        min_interaction_pae = torch.min(
            pae + (1 - target_designchain_inter_mask) * 100000
        )[None]

        is_design_token = feats["design_mask"].float()
        target_design_inter_mask = token_pad_pair_mask * (
            is_design_token[:, :, None] * is_target_token[:, None, :]
            + is_target_token[:, :, None] * is_design_token[:, None, :]
        )
        interaction_pae = (pae * target_design_inter_mask).sum(dim=(1, 2)) / (
            target_design_inter_mask.sum(dim=(1, 2)) + 1e-5
        )
        min_design_to_target_pae = torch.min(
            pae + (1 - target_design_inter_mask) * 100000
        )[None]

        out_dict = dict(
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
            interaction_pae=interaction_pae,
            min_design_to_target_pae=min_design_to_target_pae
            if feats["design_mask"].sum() > 0
            else torch.tensor([torch.nan]),
            min_interaction_pae=min_interaction_pae,
        )

        # weighted
        out_dict["protein_pae_value_weighted"] = protein_pae_value_weighted
        out_dict["ligand_pae_value_weighted"] = ligand_pae_value_weighted
        out_dict["interface_pae_value_weighted"] = interface_pae_value_weighted
        # unweighted
        out_dict["protein_pae_value_unweighted"] = protein_pae_value_unweighted
        out_dict["ligand_pae_value_unweighted"] = ligand_pae_value_unweighted
        out_dict["interface_pae_value_unweighted"] = interface_pae_value_unweighted

        if pair_dev is None:
            out_dict["pde_logits"] = pde_logits
            out_dict["pae_logits"] = pae_logits
        out_dict["pae"] = pae

        try:
            (
                ptm,
                iptm,
                ligand_iptm,
                protein_iptm,
                pair_chains_iptm,
                design_to_target_iptm,
                design_residue_iptm,
                design_iptm,
                design_iiptm,
                target_ptm,
                design_ptm,
                design_ipsae_min,
                design_to_target_ipsae,
                target_to_design_ipsae,
                chain_pair_ipsae
            ) = compute_ptms(
                pae_logits, x_pred, feats, multiplicity,
                tm_expected_value=None if pair_dev is None else pair_dev["tm"],
                expected_pae=None if pair_dev is None else pae,
            )
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm
            out_dict["design_to_target_iptm"] = design_to_target_iptm
            out_dict["design_residue_iptm"] = design_residue_iptm
            out_dict["design_iptm"] = design_iptm
            out_dict["design_iiptm"] = design_iiptm
            out_dict["target_ptm"] = target_ptm
            out_dict["design_ptm"] = design_ptm
            out_dict["design_ipsae_min"] = design_ipsae_min
            out_dict["design_to_target_ipsae"] = design_to_target_ipsae
            out_dict["target_to_design_ipsae"] = target_to_design_ipsae
            out_dict["chain_pair_ipsae"] = chain_pair_ipsae
        except Exception as e:
            print(f"Error in compute_ptms: {e}")
            out_dict["ptm"] = torch.zeros_like(complex_plddt)
            out_dict["iptm"] = torch.zeros_like(complex_plddt)
            out_dict["ligand_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["protein_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["pair_chains_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_to_target_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_residue_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_iiptm"] = torch.zeros_like(complex_plddt)
            out_dict["target_ptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_ptm"] = torch.zeros_like(complex_plddt)
            out_dict["design_ipsae_min"] = torch.zeros_like(complex_plddt)
            out_dict["design_to_target_ipsae"] = torch.zeros_like(complex_plddt)
            out_dict["target_to_design_ipsae"] = torch.zeros_like(complex_plddt)
            out_dict["chain_pair_ipsae"] = torch.zeros_like(complex_plddt)

        return out_dict

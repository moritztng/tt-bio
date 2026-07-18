"""ODesign (AF3-family all-atom binder design) on Tenstorrent -- inference port.

ODesign (github.com/OTeam-AI4S/ODesign, Apache-2.0) is the Protenix-v2 / AF3 family:
trunk Pairformer (c_s=384, c_z=128, 48 blocks, 16 heads, c_s_inputs=453) + diffusion DiT
(c_token=768, 24 blocks, 16 heads) + 3-block atom enc/dec (c_atom=128, c_atompair=16),
EDM sampler (200 steps x N_sample, N_cycle=10, bf16). The genuinely-new pieces vs
Protenix-v2 are: (a) the constraint/hotspot distogram embedder (2 Pairformer-conditioned
blocks, c=64) in the trunk, (b) the OInvFold inverse-folding head, (c) the `design` job
path. The default `prot_binding_prot` config loads ProteinMPNN (already ported in
tt_bio.proteinmpnn) for the sequence leg, so OInvFold is a stretch goal.

Pass-3 scope = the DENOISER-PARITY leg only. The ODesign DiffusionModule is structurally
identical to Protenix-v2's (same c_token=768 / 24 blocks / 16 heads / head_dim=48 /
c_atom=128; only c_z=128 vs 256 and c_s_inputs=453 vs 449 differ, both absorbed by
weight-driven linears), so this module REUSES tt_bio.protenix.DiffusionModule +
AtomFeaturization loaded with ODesign weights. The single behavioral diff in the atom
featurization is ref_element being 129-dim (vs 128); the cond builder here applies that.
Trunk conditioning (s_trunk / s_inputs / z_trunk) is taken from the captured golden
intermediates, so the denoiser is validated independently of the (unported) trunk.

DEFERRED (not built this pass, not parity-claimed): the ODesign trunk (PairformerStack +
MSAModule + the constraint/hotspot distogram embedder), the OInvFold head, the
`tt-bio design --model odesign` CLI, per-model docs, and warm single-card throughput.
The constraint embedder is only on the trunk path; it is NOT exercised by the denoiser
parity replay (which consumes precomputed trunk conditioning), so its absence does not
affect the parity number reported here.
"""
import torch
import ttnn

from .tenstorrent import get_device, CORE_GRID_MAIN
from .protenix import DiffusionModule, AtomFeaturization, _window_q, _window_kv
from . import protenix_weights as PW


class ODesign:
    """ODesign denoiser-parity harness on Tenstorrent (pass-3 scope: diffusion leg only).

    Reuses tt_bio.protenix.DiffusionModule (loaded with ODesign weights) for the per-step
    denoise, and tt_bio.protenix.AtomFeaturization for the t-independent atom single (c_l)
    and atom-pair (p_lm) conditioning. The cond builder mirrors Protenix.fold's assembly but
    reads the trunk conditioning (s_trunk / s_inputs / z_trunk) from the captured golden
    intermediates instead of running the (unported) trunk, and uses ODesign's 129-dim
    ref_element in the atom featurization f_in.

    build_cond(pre) -> cond dict; denoise_step(x_noisy, t_hat, cond) -> denoised coords;
    replay_trajectory(pre, traj) -> per-step PCC list. See scripts/odesign_traj_replay.py.
    """

    C_S, C_Z, C_S_INPUTS = 384, 128, 453
    C_ATOM, C_ATOMPAIR, C_TOKEN = 128, 16, 768
    NQ, NK, PAD_LEFT = 32, 128, 48
    R_MAX, S_MAX = 32, 2

    def __init__(self, model_state_dict, compute_kernel_config, device=None):
        self._w = model_state_dict
        self.compute_kernel_config = compute_kernel_config
        self.dev = device or get_device()

        def under(pfx):
            return {k[len(pfx):]: v for k, v in self._w.items() if k.startswith(pfx)}
        self.diffusion = DiffusionModule(under("diffusion_module."), self.dev, compute_kernel_config)
        self.diff_feat = AtomFeaturization(under("diffusion_module.atom_attention_encoder."),
                                          compute_kernel_config)

    @classmethod
    def load_from_checkpoint(cls, path, compute_kernel_config=None, device=None):
        """Load an ODesign .pt checkpoint and build the denoiser. weights_only=True."""
        dev = device or get_device()
        ckc = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        ck = ck.get("model", ck)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
        return cls(sd, ckc, dev)

    # --- upload helpers (mirror protenix._KeyedWeights; the diffusion module owns its own
    #     weight cache, these are for the cond-building linears on this object) ---
    def _tt(self, x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)

    @staticmethod
    def _to_host(t, shape=None):
        h = torch.Tensor(ttnn.to_torch(t)).float()
        return h.reshape(shape) if shape is not None else h

    def _lin(self, x, wkey):
        w = self._w[wkey]
        wtt = ttnn.from_torch(w.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                              device=self.dev, dtype=ttnn.bfloat16)
        return ttnn.linear(x, wtt, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)

    def _ln(self, x, wkey):
        w = ttnn.from_torch(self._w[wkey], layout=ttnn.TILE_LAYOUT,
                            device=self.dev, dtype=ttnn.bfloat16)
        return ttnn.layer_norm(x, weight=w, epsilon=1e-5,
                              compute_kernel_config=self.compute_kernel_config)

    @staticmethod
    def _generate_relp(feats, r_max=32, s_max=2):
        """RelativePositionEncoder feature (reference embedders.generate_relp): one-hot of
        clipped residue/token/chain offsets + same-entity. dims 2(r_max+1)+2(r_max+1)+1+
        2(s_max+1) = 139. Identical to Protenix-v2 (same r_max/s_max)."""
        import torch.nn.functional as F
        asym = feats["asym_id"].long(); res = feats["residue_index"].long()
        ent = feats["entity_id"].long(); tok = feats["token_index"].long(); sym = feats["sym_id"].long()
        sc = (asym[:, None] == asym[None, :]).long()
        sr = (res[:, None] == res[None, :]).long()
        se = (ent[:, None] == ent[None, :]).long()
        d_res = torch.clip(res[:, None] - res[None, :] + r_max, 0, 2 * r_max) * sc + (1 - sc) * (2 * r_max + 1)
        d_tok = torch.clip(tok[:, None] - tok[None, :] + r_max, 0, 2 * r_max) * sc * sr + (1 - sc * sr) * (2 * r_max + 1)
        d_ch = torch.clip(sym[:, None] - sym[None, :] + s_max, 0, 2 * s_max) * se + (1 - se) * (2 * s_max + 1)
        return torch.cat([F.one_hot(d_res, 2 * (r_max + 1)), F.one_hot(d_tok, 2 * (r_max + 1)),
                          se[..., None], F.one_hot(d_ch, 2 * (s_max + 1))], dim=-1).float()

    @staticmethod
    def _atom_pair_feats(ref_pos, ref_space_uid):
        """Algorithm 5 lines 1-3 (reference update_input_feature_dict): windowed atom-pair
        feats from ref_pos + ref_space_uid (NQ=32, NK=128, pad_left=48). Identical to
        Protenix-v2. Returns d_lm (nb,NQ,NK,3), v_lm (nb,NQ,NK,1), mask_trunked (nb,NQ,NK)."""
        import torch.nn.functional as F
        N = ref_pos.shape[0]; NQ, NK, PADL = 32, 128, 48
        nb = (N + NQ - 1) // NQ; NP = nb * NQ; qpad = NP - N
        ruid = ref_space_uid.long()
        qpos = F.pad(ref_pos.float(), (0, 0, 0, qpad)).reshape(nb, NQ, 3)
        quid = F.pad(ruid, (0, qpad), value=0).reshape(nb, NQ)
        pad_right = int((nb - 0.5) * NQ + NK / 2 - N + 0.5)
        kpos_p = F.pad(ref_pos.float(), (0, 0, PADL, pad_right))
        kuid_p = F.pad(ruid, (PADL, pad_right), value=0)
        kpos = torch.stack([kpos_p[b * NQ:b * NQ + NK] for b in range(nb)], 0)
        kuid = torch.stack([kuid_p[b * NQ:b * NQ + NK] for b in range(nb)], 0)
        d_lm = qpos[:, :, None, :] - kpos[:, None, :, :]
        v_lm = (quid[:, :, None] == kuid[:, None, :]).float().unsqueeze(-1)
        qidx = torch.arange(NP).reshape(nb, NQ); qval = (qidx < N).float()
        kglob = torch.stack([torch.arange(b * NQ - PADL, b * NQ - PADL + NK) for b in range(nb)], 0)
        kval = ((kglob >= 0) & (kglob < N)).float()
        mask_trunked = qval[:, :, None] * kval[:, None, :]
        return d_lm, v_lm, mask_trunked

    def _diffusion_pair_cond(self, z_trunk_tt, relp):
        """DiffusionConditioning pair branch (reference diffusion_module.diffusion_conditioning):
        zc = LN(concat[z_trunk, relpe(relp)]); pz = linear_z(zc); pz += transition_z1 +
        transition_z2. Identical to Protenix-v2 (no z_trunk compression -- ODesign keeps
        c_z_pair_diffusion == c_z == 128). Returns conditioned pair_z host (NT,NT,c_z)."""
        from .tenstorrent import Transition
        C = "diffusion_module.diffusion_conditioning."
        relpe = ttnn.linear(self._tt(relp), self._tt(self._w[C + "relpe.linear_no_bias.weight"].t().contiguous()),
                            compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        z_trunk_tt = ttnn.reshape(z_trunk_tt, (relpe.shape[0], relpe.shape[1], -1))
        zc = ttnn.concat([z_trunk_tt, relpe], dim=-1)
        zc = ttnn.layer_norm(zc, weight=self._tt(self._w[C + "layernorm_z.weight"]), epsilon=1e-5,
                             compute_kernel_config=self.compute_kernel_config)
        pz = ttnn.linear(zc, self._tt(self._w[C + "linear_no_bias_z.weight"].t().contiguous()),
                         compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        N = relpe.shape[0]
        pz = ttnn.reshape(pz, (1, N, N, pz.shape[-1]))
        for nm in ("transition_z1", "transition_z2"):
            sub = {k[len(C + nm + "."):]: v for k, v in self._w.items() if k.startswith(C + nm + ".")}
            pz = ttnn.add(pz, Transition(PW.remap_transition(sub), self.compute_kernel_config)(pz))
        return self._to_host(pz)

    def _plm_z_term(self, pair_z, a2t, nb, nq, nk):
        """broadcast_token_to_local_atom_pair: W_z(LN_z(z_trunk)) gathered into windowed
        atom-pair blocks (nb,nq,nk,16). The diffusion atom-encoder's p_lm cache adds this
        trunk-pair-z term (reference transformer.py prepare_cache, r_l path). Identical to
        Protenix-v2."""
        import torch.nn.functional as F
        E = "diffusion_module.atom_attention_encoder."
        lnz = F.layer_norm(pair_z, (pair_z.shape[-1],)) * self._w[E + "layernorm_z.weight"]
        ztok = F.linear(lnz, self._w[E + "linear_no_bias_z.weight"])
        N = a2t.shape[0]; NQ, NK, PADL = 32, 128, 48; NP = nb * NQ
        aq = torch.cat([a2t, torch.zeros(NP - N, dtype=torch.long)]).reshape(nb, NQ)
        ak_src = torch.cat([torch.zeros(PADL, dtype=torch.long), a2t,
                            torch.zeros(PADL + NP + NK, dtype=torch.long)])
        ak = torch.stack([ak_src[b * NQ:b * NQ + NK] for b in range(nb)], 0)
        return torch.stack([ztok[aq[b][:, None].expand(NQ, NK), ak[b][None, :].expand(NQ, NK)]
                            for b in range(nb)], 0)

    def build_cond(self, pre):
        """Assemble the denoiser cond dict from the captured golden trunk conditioning.

        pre: dict with s_trunk (NT,c_s), s_inputs (NT,c_s_inputs), z_trunk (NT,NT,c_z),
        input_data (atom feats: ref_pos, ref_space_uid, ref_element, ref_mask,
        ref_atom_name_chars, ref_charge, atom_to_token_idx, ...).
        Returns cond = {s_trunk, s_inputs, pair_z, c_l, p_lm, S, mask_trunked} with host
        tensors, matching the contract of protenix.DiffusionModule.denoise. No dit_z is
        set, so denoise uses the host fp32 DiT path (strict per-step parity, matching
        scripts/protenix_traj_replay.py)."""
        import torch
        s_inputs = pre["s_inputs"].float(); s_trunk = pre["s_trunk"].float()
        z_trunk = pre["z_trunk"].float()
        feat = pre["input_data"]
        N = feat["ref_pos"].shape[0]; NT = s_inputs.shape[0]
        a2t = feat["atom_to_token_idx"].long()
        # atom single feat f_in: cat([ref_mask(1), ref_element(129), ref_atom_name_chars(256)]) = 386
        # (ODesign ref_element is 129-dim vs Protenix-v2's 128 -- the only atom-feat diff.)
        f_in = torch.cat([feat["ref_mask"].reshape(N, 1),
                         feat["ref_element"].reshape(N, 129),
                         feat["ref_atom_name_chars"].reshape(N, 256)], dim=-1).float()
        d_lm, v_lm, mt = self._atom_pair_feats(feat["ref_pos"], feat["ref_space_uid"])
        nb, nq, nk, _ = d_lm.shape
        M = nb * nq * nk
        d = d_lm.reshape(M, 3); v = v_lm.reshape(M, 1)
        invd = (1.0 / (1.0 + (d_lm ** 2).sum(-1, keepdim=True))).reshape(M, 1)
        S = torch.zeros(N, NT); S[torch.arange(N), a2t] = 1.0
        ref_charge_asinh = torch.arcsinh(feat["ref_charge"]).reshape(N, 1).float()
        tt = self._tt
        # c_l (atom single conditioning) and p_lm (windowed atom-pair) via the diffusion
        # atom encoder's AtomFeaturization, then augment with the c_l_q/c_l_k + small_mlp
        # and the trunk-pair-z broadcast (mirrors protenix.fold's p_lm assembly).
        c_l = self._to_host(self.diff_feat.c_l(
            tt(feat["ref_pos"].float()), tt(ref_charge_asinh),
            tt(feat["ref_mask"].reshape(N, 1).float()), tt(f_in)), (N, self.C_ATOM))
        p_lm = self._to_host(self.diff_feat.p_lm(tt(d), tt(v), tt(invd), tt(mt.reshape(-1, 1).float())),
                             (nb, nq, nk, self.C_ATOMPAIR))
        pair_z = self._diffusion_pair_cond(tt(z_trunk), self._generate_relp(feat)).reshape(NT, NT, self.C_Z)
        p_lm = p_lm + self._plm_z_term(pair_z, a2t, nb, nq, nk)
        return {"s_trunk": s_trunk, "s_inputs": s_inputs, "pair_z": pair_z, "c_l": c_l,
                "p_lm": p_lm, "S": S, "mask_trunked": mt.float()}

    def denoise_step(self, x_noisy, t_hat, cond):
        """One denoise network step, matching ODesign's DiffusionModule.forward. x_noisy
        (1,N,3) is the ODesign checkpoint's ALREADY-c_in-scaled noisy coords
        (c_in = 1/sqrt(sd^2+t^2); see schedulers.add_noise_with_condition -- magnitude ~1,
        NOT raw coords). t_hat (1,), cond from build_cond. Returns the raw network output
        x_update (1,N,3) host tensor -- NOT EDM-preconditioned.

        Two conventions differ from protenix.DiffusionModule.denoise, both absorbed here
        by calling _denoise_net directly (no c_in re-scaling, no EDM preconditioning):
          (1) c_in: ODesign stores x_noisy already c_in-scaled and feeds it straight to the
              atom encoder (r_l = x_noisy); protenix.denoise expects RAW x_noisy and re-applies
              c_in internally. Feeding ODesign's scaled x_noisy to protenix.denoise would
              double-scale the encoder's coordinate input (~sigma/sigma_data too small).
          (2) EDM: ODesign's DiffusionModule.forward returns the raw network output x_update
              (the sampler's update_with_condition applies EDM later); protenix.denoise applies
              EDM itself and returns c_skip*x + c_out*x_update. The golden trajectory's
              `denoised` field is the hooked diffusion_module.forward return = x_update, so we
              compare x_update-to-x_update (no EDM)."""
        self.diffusion._atom_cond(cond)
        return self.diffusion._denoise_net(x_noisy[0].float(), t_hat, cond)

    def replay_trajectory(self, pre, traj, n_steps=None, verbose=True):
        """Replay the golden denoiser trajectory step-by-step and report REAL per-step PCC
        vs the reference `denoised`. Mirrors scripts/protenix_traj_replay.py. n_steps=None
        replays all 200; an int caps it (e.g. 1 for a single-step smoke). Returns
        (pccs, maxerrs) lists."""
        cond = self.build_cond(pre)
        N = pre["input_data"]["ref_pos"].shape[0]
        steps = traj["steps"][:n_steps] if n_steps is not None else traj["steps"]

        def pcc(u, v):
            u = u.flatten().double(); v = v.flatten().double()
            return float(((u - u.mean()) * (v - v.mean())).sum()
                         / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))
        pccs, maxerrs = [], []
        for i, st in enumerate(steps):
            xn = st["x_noisy"].float(); th = st["t_hat"].float(); ref = st["denoised"].float()
            out = self.denoise_step(xn, th, cond)
            p = pcc(out, ref[:, :N]); m = float((out - ref[:, :N]).abs().max())
            pccs.append(p); maxerrs.append(m)
            if verbose:
                print("step %3d  t_hat=%9.4g  denoised PCC %.5f  maxerr %.3e"
                      % (i, float(th.max()), p, m), flush=True)
        if verbose and pccs:
            print("\nALL-STEP denoiser PCC: min %.5f  mean %.5f  (across t_hat %.3g..%.3g)"
                  % (min(pccs), sum(pccs) / len(pccs),
                     float(steps[-1]["t_hat"].max()), float(steps[0]["t_hat"].max())), flush=True)
        return pccs, maxerrs

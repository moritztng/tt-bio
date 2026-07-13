"""P10: extend ~/of3_ref_out.pkl with the OF3 reference confidence heads
(AuxiliaryHeadsAllAtom, AF3 Algorithm 31) forward on the real ubiquitin batch, so the
device OF3ConfidenceHead can be PCC-gated per output head (PAE/PDE/pLDDT/distogram/
experimentally_resolved) against a real-weights, real-inputs golden.

Inputs are reused from already-captured golden legs (no recompute of the trunk or
diffusion): si_input/si_trunk/zij_trunk/token_mask from ``trunk_real``, the diffusion
sampler's ``xl_final`` from ``sample_diffusion_rollout_real`` as
``atom_positions_predicted``, and the feature batch from ``input_embedder_real["in"]``.
``use_zij_trunk_embedding=True`` (the reference's eval-mode value).

The reference ``AuxiliaryHeadsAllAtom`` runs entirely in fp32 (the model wraps the call
in ``torch.amp.autocast(cuda, fp32)`` at eval), so the golden head logits are fp32. The
device port isolates the 4-block confidence Pairformer as the only bf16 stage; the
z-embedding and the five heads are host-fp32, mirroring Protenix-v2's ConfidenceHead
discipline.

The reference bookkeeping ops (representative-atom selection, broadcast of the
max-atoms-per-token mask) are precomputed here with the reference helpers and stored, so
the device module consumes ready host tensors (same pattern as Protenix's precomputed
``atom_to_tokatom_idx``).

Forward-hooks ``aux_heads.pairformer_embedding`` to also capture the confidence
Pairformer output (si_conf, zij_conf) for sub-component PCC gating.

Adds key ``confidence_heads_real``:
  use_zij_trunk_embedding: True
  si_input:        [N, 449]
  si_trunk:        [N, 384]
  zij_trunk:       [N, N, 128]
  atom_positions_predicted: [N_atom, 3]
  repr_x_pred:     [N, 3]            # representative atom coords per token
  repr_x_mask:     [N]               # representative atom mask per token
  max_atom_per_token_mask: [N * 23]  # broadcast of token_mask to atom slots
  si_conf:         [N, 384]          # confidence Pairformer single output
  zij_conf:        [N, N, 128]       # confidence Pairformer pair output
  plddt_logits:                [N_atom, 50]
  experimentally_resolved_logits: [N_atom, 2]
  pae_logits:      [N, N, 64]
  pde_logits:      [N, N, 64]
  distogram_logits: [N, N, 64]

Run with the CPU reference venv, NOT the tt-bio device env:
    /tmp/of3-venv/bin/python scripts/of3_confidence_golden.py
"""
import os, sys, pickle, collections.abc
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
USE_ZIJ = True  # reference eval-mode value


def _strip(o):
    if isinstance(o, torch.Tensor):
        return o
    if (isinstance(o, (dict, collections.abc.Mapping))
            or (hasattr(o, "items") and callable(getattr(o, "items")) and hasattr(o, "__getitem__"))):
        return {k: _strip(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_strip(v) for v in o)
    return o


def main():
    from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.utils.atomize_utils import (
        get_token_representative_atoms,
        broadcast_token_feat_to_atoms,
    )

    inter = pickle.load(open(GOLD, "rb"))["intermediates"]
    ie = inter["input_embedder_real"]
    tr = inter["trunk_real"]
    xl_final = inter["sample_diffusion_rollout_real"]["xl_final"]

    si_input = tr["s_input"]          # [N, 449]
    si_trunk = tr["s_trunk"]          # [N, 384]
    zij_trunk = tr["z_trunk"]         # [N, N, 128]
    token_mask = tr["token_mask"]     # [N]

    b = ie["in"]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in b.items()}
    atom_mask = batch["atom_mask"]    # [1, N_atom]
    n_atom = int(atom_mask.shape[1])
    assert xl_final.shape[0] == n_atom, (xl_final.shape, n_atom)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    asd = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    ah = AuxiliaryHeadsAllAtom(config=C.architecture.heads).eval()
    missing, unexpected = ah.load_state_dict(asd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    # Representative atoms per token + max-atoms-per-token mask (reference helpers).
    repr_x_pred, repr_x_mask = get_token_representative_atoms(
        batch=batch, x=xl_final.unsqueeze(0), atom_mask=atom_mask)
    max_atom_per_token_mask = broadcast_token_feat_to_atoms(
        token_mask=token_mask.unsqueeze(0),
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=token_mask.unsqueeze(0),
        max_num_atoms_per_token=C.architecture.heads["max_atoms_per_token"],
    )

    # Forward-hook the confidence Pairformer to capture (si_conf, zij_conf).
    pf_log: dict = {}

    def pf_post(_m, _args, _kwargs, out):
        si_conf, zij_conf = out
        pf_log["si_conf"] = si_conf.detach().clone()
        pf_log["zij_conf"] = zij_conf.detach().clone()

    ah.pairformer_embedding.register_forward_hook(pf_post, with_kwargs=True)

    # Pre-hook the inner PairFormerStack to capture its (si, zij) inputs -- the
    # post-embed z and the si fed to the confidence pairformer, for z-embed bisect.
    def pfs_pre(_m, args, kwargs):
        si, zij = args[0], args[1]
        pf_log["si_pf_in"] = si.detach().clone()
        pf_log["zij_pf_in"] = zij.detach().clone()

    ah.pairformer_embedding.pairformer_stack.register_forward_pre_hook(pfs_pre, with_kwargs=True)

    # Per-block trajectory: hook each PairFormerStack block to capture (si, zij) in/out,
    # for block-by-block device bisect of the confidence Pairformer s/z tracks.
    blk_log: list = []

    def make_blk_hooks(idx):
        def pre(_m, args, kwargs):
            blk_log.append({"si_in": args[0].detach().clone(),
                            "zij_in": args[1].detach().clone()})
        def post(_m, _args, _kwargs, out):
            si_o, zij_o = out
            blk_log[-1]["si_out"] = si_o.detach().clone()
            blk_log[-1]["zij_out"] = zij_o.detach().clone()
        return pre, post

    for i, blk in enumerate(ah.pairformer_embedding.pairformer_stack.blocks):
        pre, post = make_blk_hooks(i)
        blk.register_forward_pre_hook(pre, with_kwargs=True)
        blk.register_forward_hook(post, with_kwargs=True)

    output = {
        "si_trunk": si_trunk.unsqueeze(0),
        "zij_trunk": zij_trunk.unsqueeze(0),
        "atom_positions_predicted": xl_final.unsqueeze(0),
    }
    with torch.no_grad():
        aux_out = ah(
            batch=batch,
            si_input=si_input.unsqueeze(0),
            output=output,
            use_zij_trunk_embedding=USE_ZIJ,
            _mask_trans=True,
        )

    rec = {
        "use_zij_trunk_embedding": USE_ZIJ,
        "si_input": si_input,
        "si_trunk": si_trunk,
        "zij_trunk": zij_trunk,
        "atom_positions_predicted": xl_final,
        "repr_x_pred": repr_x_pred[0],
        "repr_x_mask": repr_x_mask[0],
        "max_atom_per_token_mask": max_atom_per_token_mask[0],
        "si_conf": pf_log["si_conf"][0],
        "zij_conf": pf_log["zij_conf"][0],
        "si_pf_in": pf_log["si_pf_in"][0],
        "zij_pf_in": pf_log["zij_pf_in"][0],
        "block_trajectory": [
            {"si_in": b["si_in"][0], "zij_in": b["zij_in"][0],
             "si_out": b["si_out"][0], "zij_out": b["zij_out"][0]} for b in blk_log
        ],
        "plddt_logits": aux_out["plddt_logits"][0],
        "experimentally_resolved_logits": aux_out["experimentally_resolved_logits"][0],
        "pae_logits": aux_out["pae_logits"][0],
        "pde_logits": aux_out["pde_logits"][0],
        "distogram_logits": aux_out["distogram_logits"][0],
    }
    for k, v in rec.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    gold = _strip(pickle.load(open(GOLD, "rb")))
    gold["intermediates"]["confidence_heads_real"] = rec
    with open(GOLD, "wb") as f:
        pickle.dump(gold, f)
    print("wrote confidence_heads_real to", GOLD)


if __name__ == "__main__":
    main()

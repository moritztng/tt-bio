#!/usr/bin/env python3
"""Decisive parity gate for BoltzGen's diffusion_batch_size multiplicity batching.

BoltzGen's diffusion_batch_size batches N diffusion SAMPLES OF THE SAME conditioning
(one trunk pass, one masked scaffold, replicated across the batch dim) -- mechanically
the Boltz-2 "multiplicity" path (same structure, N samples), which Boltz-2 itself
proved bit-exact/lossless (see boltz2-throughput-loop memory). This harness tests
whether that lossless property transfers to BoltzGen's own AtomDiffusion/
TTDiffusionModule device path. Verdict: it does NOT -- see docs/boltzgen-batch-
threshold-ceiling.md.

Reuses the real AtomDiffusion.sample() verbatim and monkeypatches torch.randn so slot
i always draws from its own dedicated generator, regardless of batch size. So slot 0
of a multiplicity=2 batch gets an IDENTICAL noise trajectory to a standalone
multiplicity=1 run. Any final-coord difference is then purely on-device
batch-size-dependent kernel numerics (not RNG-stream confounding -- see the
e2e_mult_parity.py dead end in the boltz2-throughput-loop memory).

  TT_VISIBLE_DEVICES=2 PYTHONPATH=$PWD python3 scripts/boltzgen_batch_parity.py
"""
import os
import tempfile
from pathlib import Path
import torch

torch.set_grad_enabled(False)
STEPS = int(os.environ.get("SAMPLING_STEPS", "30"))
SEEDS = [100, 200, 300]

# ---- per-slot RNG monkeypatch (identical technique to parity_batched.py) -----
_orig_randn = torch.randn
_gens = None
def _patched_randn(*size, **kw):
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        shape = tuple(int(s) for s in size[0])
    else:
        shape = tuple(int(s) for s in size)
    gens = _gens
    if gens is not None and len(shape) >= 1 and shape[0] == len(gens):
        dtype = kw.get("dtype", None)
        device = kw.get("device", None)
        rows = [_orig_randn((1, *shape[1:]), generator=gens[i], dtype=dtype) for i in range(shape[0])]
        out = torch.cat(rows, dim=0)
        return out.to(device) if device is not None else out
    return _orig_randn(*size, **kw)
torch.randn = _patched_randn

from tt_bio.boltzgen.adapter import load_boltz_checkpoint
from tt_bio.boltzgen.task.predict.data_from_yaml import FromYamlDataModule, DataConfig
from tt_bio.boltzgen.data.tokenizer import Tokenizer
from tt_bio.boltzgen.data.featurizer import Featurizer

CACHE = Path.home() / ".boltz" / "boltzgen"
CKPT = CACHE / "boltzgen1_diverse.ckpt"
MOLDIR = str(CACHE / "mols.zip")
REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed-length spec: the real binder.yaml samples a random length per __getitem__
# (80..120), which would confound conditioning across "batched vs alone" runs before
# device numerics even enter the picture. Pin one length to isolate the batching
# effect alone.
FIXED_SPEC = Path(tempfile.gettempdir()) / "boltzgen_parity_fixed_len100.yaml"
FIXED_SPEC.write_text(
    "entities:\n"
    "  - protein:\n"
    "      id: B\n"
    "      sequence: 100\n"
    "  - file:\n"
    f"      path: {REPO_ROOT / 'examples' / 'ground_truth_structures' / 'prot.cif'}\n"
    "      include:\n"
    "        - chain:\n"
    "            id: A\n"
)

cfg = DataConfig(
    moldir=MOLDIR, multiplicity=1, yaml_path=[str(FIXED_SPEC)],
    tokenizer=Tokenizer(atomize_modified_residues=False), featurizer=Featurizer(),
    backbone_only=False, atom14=True, atom37=False, design=True,
    disulfide_prob=1.0, disulfide_on=True, diffusion_samples=1,
)
dm = FromYamlDataModule(cfg, batch_size=1, num_workers=0)
dm.num_workers = 0
batch = next(iter(dm.predict_dataloader()))

model = load_boltz_checkpoint(
    str(CKPT), strict=False, map_location="cpu",
    predict_args={"recycling_steps": 3, "sampling_steps": STEPS, "diffusion_samples": 1},
)
for m in model.modules():
    if hasattr(m, "reset_static_cache"):
        m.reset_static_cache()

sm = model.structure_module

# ---- capture the real diffusion conditioning by intercepting sample() --------
feat_masked = model.masker(batch)
_orig_sample = sm.sample
cap = {}
class _Stop(Exception):
    pass
def _capture(**kw):
    cap.update(kw)
    raise _Stop()
sm.sample = _capture
try:
    model(feat_masked, recycling_steps=3, num_sampling_steps=STEPS, diffusion_samples=1)
except _Stop:
    pass
sm.sample = _orig_sample
dc = cap["diffusion_conditioning"]
print(f"[capture] Na={cap['atom_mask'].shape[1]} steps={STEPS}", flush=True)


def run(multiplicity, gens):
    """Conditioning (s_trunk/s_inputs/feats/diffusion_conditioning) is passed
    UNBATCHED (batch=1) exactly as production forward() does -- the device
    score model broadcasts shared conditioning against `multiplicity` diffusion
    samples internally. Duplicating conditioning tensors here would instead
    exercise the (unrelated, proven-lossy) distinct-per-slot-conditioning path."""
    global _gens
    for m in model.modules():
        if hasattr(m, "reset_static_cache"):
            m.reset_static_cache()
    _gens = gens
    out = sm.sample(
        atom_mask=cap["atom_mask"],
        num_sampling_steps=STEPS,
        multiplicity=multiplicity,
        step_scale=cap.get("step_scale"),
        noise_scale=cap.get("noise_scale"),
        s_trunk=cap["s_trunk"], s_inputs=cap["s_inputs"], feats=cap["feats"],
        diffusion_conditioning=dc,
    )
    _gens = None
    return out["sample_atom_coords"].float().cpu()


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0, keepdim=True), Q - Q.mean(0, keepdim=True)
    U, _, Vt = torch.linalg.svd(Pc.T @ Qc)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d]))
    return torch.sqrt(((Pc @ (Vt.T @ D @ U.T).T - Qc) ** 2).sum(-1).mean()).item()


def g(seed):
    return torch.Generator().manual_seed(seed)


alone = {s: run(1, [g(s)])[0] for s in SEEDS[:2]}
alone_rerun = run(1, [g(SEEDS[0])])[0]
batch2 = run(2, [g(SEEDS[0]), g(SEEDS[1])])

det_raw = (alone_rerun - alone[SEEDS[0]]).abs().max().item()
print(f"\n[CONTROL] standalone(mult=1) run twice at same seed: raw_maxdiff={det_raw:.4e} A "
      f"(nonzero => device diffusion itself isn't bit-deterministic, independent of batching)", flush=True)

print("\n[RESULT] identical-noise batched(mult=2) vs standalone(mult=1):", flush=True)
for i, s in enumerate(SEEDS[:2]):
    raw = (batch2[i] - alone[s]).abs().max().item()
    rms = kabsch_rmsd(batch2[i], alone[s])
    print(f"  slot{i} (seed {s}): raw_maxdiff={raw:.4e} A   Kabsch_RMSD={rms:.4e} A", flush=True)
print(f"  (sanity) slot0 vs slot1 alone  Kabsch_RMSD={kabsch_rmsd(alone[SEEDS[0]], alone[SEEDS[1]]):.4e} A  (different seeds -> should be large)", flush=True)

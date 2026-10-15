#!/usr/bin/env python3
"""RFD3 fast-mode precision screen: one arm per process, page fixture, warm timing + CIFs kept.

    gap_screen.py ARM OUTDIR [STEPS] [SEEDS] [--fixture=perf/dsfix/fixtures/rfd3_R4.json] [--batch=1] [--fast=1]

--fast=1 folds through `design.run_design(fast=True)`, i.e. the fast set as a user gets it, on top of
the arm's knobs. Without it the arm is exactly what its name says and nothing else, which is what an
A/B wants; with it the run answers "is this what --fast ships today".

--batch=N folds N designs per seed in one batch; design i of seed S is kept as <arm>_s<10*S+i>.cif.

ARM is a '+'-joined set of knobs applied before the denoiser is built:
  exact     the pre-SDPA shipped chain: every knob below off, whatever the defaults are
  hifi2     RFD3_HIFI2, every RFD3 compute kernel config at MathFidelity.HiFi2 (default HiFi4)
  lofi      ... at MathFidelity.LoFi
  acco      ... with fp32_dest_acc_en off (bf16 dest: twice the dest tiles, larger subblocks)
  w8        every 2-D, tile-aligned weight that Module.torch_to_tt uploads as bf16 goes up as bfloat8_b
  asdpa     RFD3_ATOM_SDPA, atom attention as one fused SDPA (on by default since 10-11)
  dsdpa     RFD3_DIT_SDPA, the dense (DiT) attention as one fused SDPA (new, off by default)
  tb8       RFD3_TRANSITION_B8, the pair Transitions' hidden activations in bfp8
  sf        RFD3_SIGMOID_FOLD, sigmoid gates folded into their multiply (on by default)
  rs        RFD3_RESIDENT_STEP, the denoise step stays on the card between modules (on by default)
  pws       RFD3_PAIR_WS, the token pair Transition as one weight-stationary program (tt_bio.pair_swiglu_ws)
  pwb8      RFD3_PAIR_WS_B8, pws with bfp8 weights at HiFi2
  gsm bs pzc fgrid   the existing off-by-default levers RFD3_GATHERED_SOFTMAX, RFD3_BLOCK_SPARSE,
            RFD3_PROCESS_Z_COLLAPSE, RFD3_FAST_GRID (declined before only for not being bit-exact)

Arms are absolute: every env knob not named is set to 0, so "exact" keeps meaning the same path
after a default flips. SEEDS is a comma list (default 42). Per seed: one discarded warmup fold for the first seed only, then
one timed fold; the CIF is copied to OUTDIR/<arm>_s<seed>.cif so accuracy is graded offline against
the exact arm's CIF of the same seed (same-seed drift) and against other seeds (the seed floor).
"""
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

ENV_KNOBS = {"gsm": "RFD3_GATHERED_SOFTMAX", "bs": "RFD3_BLOCK_SPARSE",
             "pzc": "RFD3_PROCESS_Z_COLLAPSE", "fgrid": "RFD3_FAST_GRID",
             "asdpa": "RFD3_ATOM_SDPA", "tb8": "RFD3_TRANSITION_B8",
             "dsdpa": "RFD3_DIT_SDPA", "hifi2": "RFD3_HIFI2", "sf": "RFD3_SIGMOID_FOLD",
             "rs": "RFD3_RESIDENT_STEP", "pws": "RFD3_PAIR_WS", "kd": "RFD3_ATTN_KDTREE",
             "pwb8": "RFD3_PAIR_WS_B8"}
_arm = set(sys.argv[1].split("+")) if len(sys.argv) > 1 else set()
for _k, _env in ENV_KNOBS.items():
    os.environ[_env] = "1" if _k in _arm else "0"   # read at import, so before tt_bio.rfd3 loads

sys.path.insert(0, os.getcwd())
import ttnn                                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                             # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                               # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ARM, OUT = args[0], pathlib.Path(args[1])
STEPS = int(args[2]) if len(args) > 2 else 200
SEEDS = [int(s) for s in (args[3] if len(args) > 3 else "42").split(",")]
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")
BATCH = int(opts.get("batch", 1))
FAST = opts.get("fast", "0") == "1"
knobs = set(ARM.split("+")) - {"exact"}
assert knobs <= {"lofi", "w8", "acco"} | set(ENV_KNOBS), knobs

pick = [ttnn.MathFidelity.LoFi] if "lofi" in knobs else []
if pick or "acco" in knobs:
    _ckc = M._default_compute_kernel_config


    def _ckc_fid():
        c = _ckc()
        return type(c)(math_fidelity=pick[0] if pick else c.math_fidelity,
                       math_approx_mode=c.math_approx_mode,
                       fp32_dest_acc_en=c.fp32_dest_acc_en and "acco" not in knobs,
                       packer_l1_acc=c.packer_l1_acc)
    M._default_compute_kernel_config = _ckc_fid

W8 = [0, 0]
if "w8" in knobs:
    _t2t = T.Module.torch_to_tt

    def _t2t_w8(self, key, transform=lambda x: x.t(), dtype=None):
        w = transform(self.weights[key])
        if (dtype in (None, ttnn.bfloat16) and w.ndim == 2
                and w.shape[0] % 32 == 0 and w.shape[1] % 32 == 0):
            W8[0] += 1
            return _t2t(self, key, transform, dtype=ttnn.bfloat8_b)
        W8[1] += 1
        return _t2t(self, key, transform, dtype=dtype)
    T.Module.torch_to_tt = _t2t_w8

WALL = []
_sample = RFD3Sampler.sample


def _timed(self, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, *a, **k)
    WALL.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(tag, seed):
    specs = json.loads(pathlib.Path(FIXTURE).read_text())
    out_dir = "/tmp/gap_screen_%s_%s_%d" % (ARM, tag, os.getpid())   # parallel jobs of one arm share a box
    shutil.rmtree(out_dir, ignore_errors=True)
    WALL.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=STEPS, seed=seed, num_designs=BATCH, batch_size=BATCH,
                           fast=FAST, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = "|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs) or "NO CIF"
    return WALL[0], dig, cifs


OUT.mkdir(parents=True, exist_ok=True)
rec = {"arm": ARM, "steps": STEPS, "fixture": FIXTURE, "batch": BATCH, "fast": FAST, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "host": os.uname().nodename, "folds": []}
s, d, _ = fold("warm", SEEDS[0])
rec["warm_s"] = s
print("[screen %s] warmup %.3f s %s  w8 %s" % (ARM, s, d, W8), flush=True)
for seed in SEEDS:
    s, d, cifs = fold("s%d" % seed, seed)
    for i, c in enumerate(cifs[:BATCH]):
        shutil.copy(c, OUT / ("%s_s%d.cif" % (ARM, seed if BATCH == 1 else 10 * seed + i)))
    rec["folds"].append({"seed": seed, "sampler_s": s, "digest": d})
    print("[screen %s] seed %d %.3f s %s" % (ARM, seed, s, d), flush=True)
rec["w8_counts"] = W8
rec["sdpa_calls"] = {"atom": M.ATOM_SDPA_CALLS[0], "dense": M.DIT_SDPA_CALLS[0]}
(OUT / ("%s.json" % ARM)).write_text(json.dumps(rec, indent=1))
print("wrote", OUT / ("%s.json" % ARM))

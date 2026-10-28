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
  pps       RFD3_PAIR_WS_PSILU, pws with the silu on the packer, overlapping the matmuls (bit-identical)
  dkm       RFD3_DKNN_MOVERS, the moving atoms' index rows on the card (dknn row range), static rows on the host tree
  dkst      RFD3_DKNN_STATIC, the card index's static rows against their static candidates and the movers only
  pmemo     RFD3_PAIR_MEMO, the token encoder pair path recomputed only on row tiles whose distance bins changed
  rmsb      RFD3_PF_RMSB, the pairformer SDPA mask from one rms_pair_bias program (norm weight folded into W)
  pzr       RFD3_PZ_ROWS, process_z as one gather + a + em * inv program on the pair memo's changed row tiles
  mm2d      RFD3_DIT_MM2D, the DiT linears through a pinned 2D multicast config (regroups K, graded on accuracy)
  mm2l      RFD3_DIT_MM2D_L1, with mm2d: those linears write their output to L1
  at        RFD3_ATTN_THREAD, the atom neighbour index built on a worker thread
  dpb       RFD3_DPB, the DiT SDPA masks as one program (tt_bio.dit_pair_bias)
  apb       RFD3_ATOM_PB_FUSED, one pair-bias projection for all three atom blocks (default on)
  pdpb      RFD3_PF_DPB, the token encoder's pairformer pair bias through tt_bio.dit_pair_bias
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

from gap_knobs import ENV_KNOBS, apply                                   # noqa: E402
_arm = apply(sys.argv[1] if len(sys.argv) > 1 else "")   # read at import, so before tt_bio.rfd3 loads

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
assert knobs <= {"lofi", "w8", "acco", "idxfree", "binsfree", "dkfree", "hsplit", "dprof"} | set(ENV_KNOBS), knobs

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

# idxfree / binsfree (instruments, not levers): the warmup fold records every attention index (idxfree) or token
# distogram (binsfree) keyed by its inputs, later folds replay a hit instead of building it. Run the warm seed again and
# that host work costs nothing; the digest must equal the plain arm's, and the time saved is the most any faster build
# (on device or not) can buy.
import torch                                                              # noqa: E402

IDX = [0, 0]   # replayed, built
_rec = [True]


def _replayed(fn, coords, extra):
    book = {}

    def run(*a, **kw):
        x = coords(*a)
        key = (hashlib.sha256(x.detach().float().contiguous().numpy().tobytes()).hexdigest(), tuple(x.shape)) + extra(*a)
        hit = book.get(key)
        if hit is not None and not _rec[0]:
            IDX[0] += 1
            return hit[0].to(hit[1], copy=True)
        IDX[1] += 1
        out = fn(*a, **kw)
        if _rec[0]:   # held as int32 (indices < 2^31, bins < 65): 200 steps of atom indices are ~1.9 GB that way
            book[key] = (out.to(torch.int32, copy=True), out.dtype)
        return out
    return run


if "idxfree" in knobs:
    M._create_attention_indices = _replayed(M._create_attention_indices, lambda f, X_L, tok_idx, n_keys, *r: X_L,
                                            lambda f, X_L, tok_idx, n_keys, *r: (n_keys, len(tok_idx)))
if "binsfree" in knobs:
    M._scaled_distogram_bins = _replayed(M._scaled_distogram_bins, lambda R_L, *r: R_L, lambda *a: ())
# dkfree: the same for the card atom index (dknn.Pending, which idxfree never sees): a replayed step launches no dknn
# program and reads nothing back, so the warm seed's saving is the most the card index can still cost.
if "dkfree" in knobs:
    from tt_bio import dknn as _dk
    _Pending = _dk.Pending

    class _Replay:
        book = {}

        def __init__(self, device, X_L, mask, k, static=None):
            self.key = (hashlib.sha256(X_L.detach().float().contiguous().numpy().tobytes()).hexdigest(), tuple(X_L.shape), k)
            hit = self.book.get(self.key)
            self.hit = None if _rec[0] or hit is None else hit
            self.real = _Pending(device, X_L, mask, k, static) if self.hit is None else None

        def result(self):
            if self.hit is not None:
                IDX[0] += 1
                return self.hit.clone()
            IDX[1] += 1
            out = self.real.result()
            if _rec[0]:
                self.book[self.key] = out.clone()
            return out
    _dk.Pending = _Replay

# hsplit: host seconds inside each region's call (enqueue only for device regions; no sync is added), so a region whose
# enqueue time is close to its Tracy device time is dispatch-bound. to_torch is the blocking readback (waits on the card).
HOST = {}


def _host_timed(owner, name, key):
    fn = getattr(owner, name)

    def run(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            HOST[key] = HOST.get(key, 0.0) + time.perf_counter() - t0
    setattr(owner, name, run)


if "hsplit" in knobs:
    for owner, name, key in ((M.DiffusionTokenEncoder, "run_device", "token_encoder"),
                             (M.LocalTokenTransformer, "__call__", "dit"),
                             (M.LocalAtomTransformer, "__call__", "encoder"),
                             (M.CompactStreamingDecoder, "run_full_device", "decoder"),
                             (M, "_create_attention_indices", "attn_idx_all_threads"),
                             (M, "_scaled_distogram_bins", "bins"),
                             (ttnn, "to_torch", "to_torch")):
        _host_timed(owner, name, key)

# dprof: cProfile over the device regions' host calls only (enqueue side), written per fold to <arm>.dprof.txt by
# tottime, to name the Python spent per op where an enqueue runs slower than its region's device time.
PROF = None
if "dprof" in knobs:
    import cProfile
    import pstats
    from tt_bio import dknn as _dkm
    PROF, _depth = cProfile.Profile(), [0]

    def _profiled(owner, name):
        fn = getattr(owner, name)

        def run(*a, **k):
            _depth[0] += 1
            if _depth[0] == 1:
                PROF.enable()
            try:
                return fn(*a, **k)
            finally:
                _depth[0] -= 1
                if _depth[0] == 0:
                    PROF.disable()
        setattr(owner, name, run)
    for owner, name in ((M.DiffusionTokenEncoder, "run_device"), (M.LocalTokenTransformer, "__call__"),
                        (M.LocalAtomTransformer, "__call__"), (M.CompactStreamingDecoder, "run_full_device"),
                        (_dkm.Pending, "__init__"), (_dkm.Pending, "result"), (_dkm.Static, "plan")):
        _profiled(owner, name)

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
    HOST.clear()
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
if knobs & {"idxfree", "binsfree", "dkfree"}:
    _rec[0] = False
    rec["idx_warm_built"] = IDX[1]
    IDX[:] = [0, 0]
for seed in SEEDS:
    if PROF is not None:
        PROF = cProfile.Profile()
    s, d, cifs = fold("s%d" % seed, seed)
    if PROF is not None:
        with open(OUT / ("%s.s%d.dprof.txt" % (ARM, seed)), "w") as f:
            st = pstats.Stats(PROF, stream=f).sort_stats("tottime")
            st.print_stats(60)
            st.sort_stats("cumulative").print_stats(80)
    for i, c in enumerate(cifs[:BATCH]):
        shutil.copy(c, OUT / ("%s_s%d.cif" % (ARM, seed if BATCH == 1 else 10 * seed + i)))
    rec["folds"].append({"seed": seed, "sampler_s": s, "digest": d})
    if "pmemo" in knobs:
        from tt_bio.rfd3 import model as _m
        rec["folds"][-1]["pair_memo_rows"] = list(_m.PAIRMEMO)
        _m.PAIRMEMO[:] = [0, 0]
    if "rmsb" in knobs:
        from tt_bio.rfd3 import model as _m
        rec["folds"][-1]["pf_rmsb_calls"] = _m.PF_RMSB_CALLS[0]
        _m.PF_RMSB_CALLS[0] = 0
    if "hsplit" in knobs:
        rec["folds"][-1]["host_ms_per_step"] = {k: round(1e3 * v / STEPS, 2) for k, v in HOST.items()}
        print("[screen %s] host ms/step %s" % (ARM, rec["folds"][-1]["host_ms_per_step"]), flush=True)
    print("[screen %s] seed %d %.3f s %s" % (ARM, seed, s, d), flush=True)
rec["w8_counts"] = W8
rec["idx_replayed_built"] = IDX
rec["sdpa_calls"] = {"atom": M.ATOM_SDPA_CALLS[0], "css": M.CSS_CALLS[0], "dense": M.DIT_SDPA_CALLS[0]}
rec["pf_dpb_calls"] = {"served": M.PF_DPB_CALLS[0], "declined": M.PF_DPB_CALLS[1]}
(OUT / ("%s.json" % ARM)).write_text(json.dumps(rec, indent=1))
print("wrote", OUT / ("%s.json" % ARM))

"""At the diverging softmax, compare the CACHED program against a FRESH compile in-process.

Three answers for one bit-identical input:
  cached  -- what the sequenced fold actually computes (wrong: mpro ends 0.335 A off)
  fresh   -- the same call after disable_and_clear_program_cache() in the SAME process
  isolated-- what a process that folds only mpro gets (passed in as --reference)

fresh == isolated  => the cached program differs from what this device state compiles now,
                     i.e. the program's behaviour depends on state outside its cache key.
fresh == cached    => the program is not the variable; the input bytes (tile padding) or
                     some persistent device state is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--contigs", nargs="*", default=[])
ap.add_argument("--specs", type=Path, nargs="*", default=[])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
ap.add_argument("--reference", type=float, default=None)
ap.add_argument("--dump-input", type=Path, default=None)
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

import ttnn  # noqa: E402
import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402

LAST = [False]
IN_DECODER = [0]
IN_BLOCK = [0]
DONE = [False]
_softmax = ttnn.softmax


def summarize(t):
    return round(ttnn.to_torch(t).double().sum().item(), 9)


def patched_softmax(x, *a, **kw):
    dev = get_device()
    if not (LAST[0] and IN_DECODER[0] and IN_BLOCK[0] and not DONE[0]):
        return _softmax(x, *a, **kw)
    DONE[0] = True
    entries = dev.num_program_cache_entries()
    cached = _softmax(x, *a, **kw)
    s_cached = summarize(cached)
    xin = ttnn.to_torch(x).double()
    print(f"[probe] shape={tuple(x.shape)} padded={tuple(x.padded_shape)} dtype={x.dtype} "
          f"cache_entries={entries}", flush=True)
    print(f"[probe] input  sum={xin.sum().item():.12e} absmax={xin.abs().max().item():.9f}",
          flush=True)
    if args.dump_input:
        torch.save(ttnn.to_torch(x), args.dump_input)
        print(f"[probe] dumped input to {args.dump_input}", flush=True)
    dev.disable_and_clear_program_cache()
    dev.enable_program_cache()
    fresh = _softmax(x, *a, **kw)
    s_fresh = summarize(fresh)
    print(f"[probe] cached sum={s_cached:.9f}", flush=True)
    print(f"[probe] fresh  sum={s_fresh:.9f}", flush=True)
    if args.reference is not None:
        print(f"[probe] isolat sum={args.reference:.9f}", flush=True)
        print(f"[probe] fresh==isolated: {abs(s_fresh - args.reference) < 1e-6}   "
              f"cached==isolated: {abs(s_cached - args.reference) < 1e-6}", flush=True)
    ttnn.deallocate(fresh)
    return cached


ttnn.softmax = patched_softmax


def wrap_scope(cls, meth, counter):
    orig = getattr(cls, meth)

    def patched(self, *a, **kw):
        counter[0] += 1
        try:
            return orig(self, *a, **kw)
        finally:
            counter[0] -= 1
    setattr(cls, meth, patched)


wrap_scope(R.CompactStreamingDecoder, "run_device", IN_DECODER)
wrap_scope(R.RFD3AtomBlock, "__call__", IN_BLOCK)

ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu",
                weights_only=True)
dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu",
                 weights_only=True)
dev_ti = R.build_token_initializer(ti)
dm = R.build_diffusion_module(dmw)

cases = [(c, str(PDB), {"input": str(PDB), "contig": c}) for c in args.contigs]
for spec_path in args.specs:
    data = json.loads(Path(spec_path).read_text())
    src = Path(data["input"])
    if not src.is_absolute():
        src = Path(spec_path).parent / src
    cases.append((Path(spec_path).parent.name, str(src), dict(data, input=str(src))))

for contig, pdb_path, spec_data in cases:
    last = contig == cases[-1][0]
    s = InputSpecification.from_dict(spec_data)
    s.validate()
    f = featurize(pdb_path, s)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    L = f["ref_pos"].shape[0]
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    torch.manual_seed(0)
    X1 = torch.randn(1, L, 3) * 16.0
    for D in args.batches:
        XD = X1.expand(D, -1, -1).contiguous()
        LAST[0] = last and D == 1
        with torch.no_grad():
            out = dm(X_noisy_L=XD, t=torch.full((D,), 8.0), f=f, **init)["X_L"]
        LAST[0] = False
        print(f"{contig} L={L} D={D}: sum={out.double().sum().item():.12e}", flush=True)

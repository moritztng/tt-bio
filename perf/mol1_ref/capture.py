"""Capture Mol-1 upstream goldens: run the upstream runner unchanged, with forward hooks on every module the TT
port mirrors and a recording proxy on the sampler's RNG.

    python -I capture.py --golden_root DIR -- <mol/runner/inference.py arguments>

Writes, per <sample>/seed_<seed>/:
  features.pt                 input_feature_dict as the model receives it (before the loop prunes MSA/template keys)
  <module>.call<k>.pt         {"args", "kwargs", "out"} (or {"out"}) for the calls listed in KEEP, fp32 on CPU
  sampler.pt                  per diffusion call: x_noisy, t_hat, x_denoised; plus the noise schedule
  draws.pt                    every random draw of the sampler in call order: [(tag, tensor)], tag in
                              {init, rot, trans, eps}; replaying them reproduces the trajectory exactly
  MANIFEST.json               name -> shape/dtype/sha256 of every tensor file
Runs on CPU only; the upstream clone is imported from the venv it was installed into, never from its directory.
"""
import argparse
import dataclasses
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

# module path -> (calls stored with inputs + output, calls stored output-only); "all" = every call.
# Recycle-indexed modules run once per recycle (20 by default), the diffusion module once per step x sample.
KEEP = {
    "input_embedder": ({0}, set()),
    "relative_position_encoding": ({0}, set()),
    "template_embedder": ({0}, {1, 19}),
    "msa_module": ({0}, {1, 19}),
    "pairformer_stack": ({0}, "all"),      # (s, z) after every recycle = the full trunk per recycle
    "diffusion_module.diffusion_conditioning": ({0, 100}, set()),
    "diffusion_module": ({0, 100}, set()),  # one full diffusion step golden; every step's x is in sampler.pt
    "distogram_head": ({0}, set()),
    "confidence_head": ({0}, set()),
}
# trunk blocks/adapters, first recycle only. Of the 48 pairformer blocks keep the first two, the one after the
# before_block_8 adapter, the middle and the last: block k's input is block k-1's output, and all 48 at 510 tokens
# would be ~13 GB a case.
BLOCK_RE = re.compile(r"^(pairformer_stack\.(blocks\.(0|1|8|23|47)|adapters\.[^.]+)|msa_module\.blocks\.[^.]+|"
                      r"template_embedder\.pairformer_stack\.blocks\.[^.]+)$")
METHODS = {  # (module path, method) wrapped like a forward
    ("diffusion_module.diffusion_conditioning", "prepare_cache"),
    ("diffusion_module.atom_attention_encoder", "prepare_cache"),
    ("diffusion_module", "prepare_projected_pair_biases"),
}


def detach(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):  # stored as a plain dict: loads without mol installed
        return {f.name: detach(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").clone()
    if isinstance(x, dict):
        return {k: detach(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach(v) for v in x) if not hasattr(x, "_fields") else type(x)(*[detach(v) for v in x])
    return x


def strip_features(kwargs):
    # the feature dict is stored once in features.pt; a call refers to it by name
    return {k: ("<features.pt>" if k == "input_feature_dict" else v) for k, v in kwargs.items()}


class RecordingTorch:
    """Stands in for `torch` inside the sampler modules so every randn is recorded, in order."""

    def __init__(self, log, tag_fn):
        self._log, self._tag_fn = log, tag_fn

    def randn(self, *a, **k):
        t = torch.randn(*a, **k)
        self._log.append((self._tag_fn(), t.detach().cpu().clone()))
        return t

    def __getattr__(self, name):
        return getattr(torch, name)


class Capture:
    def __init__(self, root):
        self.root = Path(root)
        self.draws = []
        self.reset()

    def reset(self):
        # cleared in place: the RNG proxies hold a reference to self.draws
        self.calls, self.counts, self.sampler, self.timeline = {}, {}, [], []
        self.draws.clear()
        self.features = None
        self.t0 = time.perf_counter()

    def want(self, name, k):
        """-> "full", "out" or None for call k of module `name`."""
        full, out = KEEP.get(name, ({0}, set()))  # trunk blocks, adapters and wrapped methods: first call in full
        if full == "all" or k in full:
            return "full"
        if out == "all" or k in out:
            return "out"
        return None

    def record(self, name, args, kwargs, out):
        k = self.counts.get(name, 0)
        self.counts[name] = k + 1
        if "." not in name:  # top-level modules: wall time since the forward started, for run planning
            self.timeline.append((name, k, round(time.perf_counter() - self.t0, 3)))
        if name == "diffusion_module":
            self.sampler.append({"x_noisy": detach(kwargs["x_noisy"]),
                                 "t_hat": detach(kwargs["t_hat_noise_level"]),
                                 "x_denoised": detach(out)})
        mode = self.want(name, k)
        if mode == "full":
            self.calls[f"{name}.call{k}"] = {"args": detach(list(args)), "kwargs": detach(strip_features(kwargs)),
                                             "out": detach(out)}
        elif mode == "out":
            self.calls[f"{name}.call{k}"] = {"out": detach(out)}

    def install(self, model):
        mods = dict(model.named_modules())
        for name, mod in mods.items():
            if name in KEEP or BLOCK_RE.match(name):
                mod.register_forward_hook(lambda m, a, kw, out, n=name: self.record(n, a, kw, out), with_kwargs=True)
        for path, meth in METHODS:
            mod = mods[path]
            orig = getattr(mod, meth)

            def wrapped(*a, _orig=orig, _n=f"{path}.{meth}", **kw):
                out = _orig(*a, **kw)
                self.record(_n, a, kw, out)
                return out
            setattr(mod, meth, wrapped)
        model.register_forward_pre_hook(self._features, with_kwargs=True)

    def _features(self, m, a, kw):
        self.reset()
        self.features = detach(kw["input_feature_dict"])

    def install_rng(self):
        import mol.model.generator as gen
        import mol.model.utils as mu

        def gen_tag():  # a generator draw right after an augmentation is that step's eps, otherwise chunk init
            tag = "eps" if self._step_open else "init"
            self._step_open = False
            return tag
        self._step_open = False
        gen.torch = RecordingTorch(self.draws, gen_tag)
        mu.torch = RecordingTorch(self.draws, lambda: "trans")
        orig_rot, orig_aug = mu.uniform_random_rotation, gen.centre_random_augmentation

        def rot(N_sample=1):
            r = orig_rot(N_sample)
            self.draws.append(("rot", r.clone()))
            return r

        def aug(*a, **k):
            self._step_open = True  # every augmentation opens a step whose next generator randn is eps
            return orig_aug(*a, **k)
        mu.uniform_random_rotation = rot
        gen.centre_random_augmentation = aug
        orig_sd = gen.sample_diffusion

        def sample_diffusion(*a, **k):
            self.noise_schedule = detach(k.get("noise_schedule"))
            return orig_sd(*a, **k)
        gen.sample_diffusion = sample_diffusion
        import mol.model.mol as mm
        mm.sample_diffusion = sample_diffusion

    def flush(self, sample, seed):
        d = self.root / sample / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        files = {"features.pt": self.features, "sampler.pt": {"calls": self.sampler,
                                                             "noise_schedule": getattr(self, "noise_schedule", None)},
                 "draws.pt": self.draws}
        files.update({f"{k}.pt": v for k, v in self.calls.items()})
        man = {}
        for fn, obj in files.items():
            p = d / fn
            torch.save(obj, p)
            man[fn] = {"bytes": p.stat().st_size, "sha256": sha256(p), "summary": summarize(obj)}
        (d / "timeline.json").write_text(json.dumps(self.timeline))
        (d / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True))
        print(f"[capture] {sample} seed {seed}: {len(files)} files -> {d}", flush=True)
        self.reset()


def summarize(obj, depth=0):
    if isinstance(obj, torch.Tensor):
        return f"{tuple(obj.shape)} {str(obj.dtype).replace('torch.', '')}"
    if depth > 2:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {str(k): summarize(v, depth + 1) for k, v in list(obj.items())[:64]}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 8:
            return f"{type(obj).__name__}[{len(obj)}] of {summarize(obj[0], depth + 1)}"
        return [summarize(v, depth + 1) for v in obj]
    return repr(obj)[:80]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden_root", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest

    import mol.runner.inference as inf
    cap = Capture(a.golden_root)
    cap.install_rng()
    orig_init_model = inf.InferenceRunner.init_model

    def init_model(self):
        orig_init_model(self)
        cap.install(self.model)
    inf.InferenceRunner.init_model = init_model

    orig_dump = None

    def dump(self_dumper, *args, **kw):
        cap.flush(kw["pdb_id"], kw["seed"])
        return orig_dump(self_dumper, *args, **kw)
    import mol.runner.dumper as dm
    orig_dump = dm.DataDumper.dump
    dm.DataDumper.dump = dump

    sys.argv = ["inference.py"] + rest
    inf.run()


if __name__ == "__main__":
    main()

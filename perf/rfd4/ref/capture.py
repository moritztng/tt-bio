"""Golden capture for upstream RFD4-Proteina (row rfd4-ref).

Registers forward hooks on the modules the tt-bio port will replace, runs a shipped
example end to end on CPU fp32, and writes one .pt per capture plus a JSONL record
per capture. `index.py` merges the JSONL files into state/rfd4/goldens/INDEX.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Module classes worth a golden: the flow transformer stack, pair bias attention,
# pair update / triangle ops, the autoencoder, and the feature factory.
DEFAULT_CLASSES = [
    "LocalLatentsTransformer",
    "MultiheadAttnAndTransition",
    "PairBiasAttention",
    "MultiHeadBiasedAttentionADALN_MM",
    "PairReprUpdate",
    "TriangleAttention",
    "TriangleMultiplication",
    "EncoderTransformer",
    "DecoderTransformer",
    "FeatureFactory",
    "FourierEmbedding",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _describe(obj, depth: int = 0):
    """Shape/dtype description of an arbitrarily nested arg tree."""
    if isinstance(obj, torch.Tensor):
        return {"t": "tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if depth > 3:
        return {"t": type(obj).__name__}
    if isinstance(obj, (list, tuple)):
        return [_describe(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {k: _describe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return {"t": type(obj).__name__}


def _to_f64(obj, depth: int = 0):
    """Cast every floating tensor in an arg tree to float64, leave the rest alone."""
    if isinstance(obj, torch.Tensor):
        return obj.double() if obj.is_floating_point() else obj
    if depth > 3:
        return obj
    if isinstance(obj, tuple):
        return tuple(_to_f64(x, depth + 1) for x in obj)
    if isinstance(obj, list):
        return [_to_f64(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_f64(v, depth + 1) for k, v in obj.items()}
    return obj


def _detach(obj, depth: int = 0):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if depth > 3:
        return None
    if isinstance(obj, (list, tuple)):
        return [_detach(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {k: _detach(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return None


class CapturesComplete(Exception):
    """Raised to end generation once every hooked module has its golden."""


class Capturer:
    """Hooks selected submodules and writes <out>/<feature>/<slug>.pt per capture."""

    def __init__(self, out_dir: Path, feature: str, seed: int, sha: str,
                 classes: list[str], per_module: int, records: Path, f64: bool = False,
                 stop_when_done: bool = False):
        self.dir = out_dir / feature
        self.dir.mkdir(parents=True, exist_ok=True)
        self.feature = feature
        self.seed = seed
        self.sha = sha
        self.classes = set(classes)
        self.per_module = per_module
        self.records = records
        self.f64 = f64
        self.stop_when_done = stop_when_done
        self.sampler_done = True  # set False by wrap_sampler while steps are still owed
        self.counts: dict[str, int] = {}
        self.sampled: dict[str, list[str]] = {}
        self.handles: list = []

    def attach(self, model: torch.nn.Module, per_class: int = 0) -> int:
        """Hook the selected classes. `per_class` > 0 samples that many module instances
        evenly across each class's ordered list (a 26-block stack costs ~9 MB a block, so
        the set is pruned by sampling blocks, never by dropping a class or a feature)."""
        by_class: dict[str, list] = {}
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if cls in self.classes:
                by_class.setdefault(cls, []).append((name, mod))
        n = 0
        for cls, items in by_class.items():
            if per_class and len(items) > per_class:
                step = (len(items) - 1) / (per_class - 1) if per_class > 1 else 1
                keep = sorted({round(i * step) for i in range(per_class)})
                items = [items[i] for i in keep]
            self.sampled[cls] = [name for name, _ in items]
            for name, mod in items:
                self.handles.append(mod.register_forward_hook(self._make_hook(name), with_kwargs=True))
                n += 1
        return n

    def detach_all(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def save_blob(self, slug: str, payload: dict, cls: str, module: str, call: int,
                  extra: dict | None = None) -> None:
        """Write one capture that did not come from a module hook (sampler state, features)."""
        path = self.dir / f"{slug}.pt"
        payload = {"feature": self.feature, "seed": self.seed, "upstream_sha": self.sha, **payload}
        torch.save(payload, path)
        rec = {
            "feature": self.feature, "module": module, "cls": cls, "call": call,
            "seed": self.seed, "upstream_sha": self.sha, "dtype": "float32", "device": "cpu",
            "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path),
            "in_args": None, "in_kwargs": _describe(payload.get("batch")),
            "out": _describe(payload.get("output")), "f64": False, "f64_error": None,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            rec.update(extra)
        with self.records.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _make_hook(self, name: str):
        def hook(mod, args, kwargs, output):
            key = f"{type(mod).__name__}:{name}"
            idx = self.counts.get(key, 0)
            if idx >= self.per_module:
                self.counts[key] = idx + 1
                return
            self.counts[key] = idx + 1
            slug = f"{key.replace(':', '__').replace('.', '_')}__call{idx}"
            path = self.dir / f"{slug}.pt"
            payload = {
                "feature": self.feature,
                "module": name,
                "cls": type(mod).__name__,
                "call": idx,
                "seed": self.seed,
                "upstream_sha": self.sha,
                "args": _detach(args),
                "kwargs": _detach(kwargs),
                "output": _detach(output),
            }
            f64_err = None
            if self.f64:
                try:
                    import copy

                    ref = copy.deepcopy(mod).double().eval()
                    with torch.no_grad():
                        out64 = ref(*_to_f64(args), **_to_f64(kwargs))
                    payload["output_f64"] = _detach(out64)
                    del ref
                except Exception as exc:  # a module the reference cannot run in float64
                    f64_err = f"{type(exc).__name__}: {exc}"[:300]
            torch.save(payload, path)
            rec = {
                "feature": self.feature,
                "module": name,
                "cls": type(mod).__name__,
                "call": idx,
                "seed": self.seed,
                "upstream_sha": self.sha,
                "dtype": "float32",
                "device": "cpu",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "in_args": _describe(args),
                "in_kwargs": _describe(kwargs),
                "out": _describe(output),
                "f64": bool(self.f64 and f64_err is None),
                "f64_error": f64_err,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with self.records.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if self.stop_when_done and self.sampler_done and len(self.counts) == len(self.handles) \
                    and all(v >= self.per_module for v in self.counts.values()):
                raise CapturesComplete

        return hook


def wrap_sampler(cap: "Capturer", model, steps: int) -> None:
    """Record the first `steps` sampler steps: the batch going into the trunk and what comes back.

    `Proteina.predict_for_sampling` is the per-step entry point; wrapping it gives the sampler
    state (x_t, time, recycle depth, conditioning) and the trunk's prediction for that step.
    """
    inner = model.predict_for_sampling
    state = {"i": 0}
    if steps:
        cap.sampler_done = False

    def wrapped(batch, mode, recycle_depth):
        out = inner(batch, mode, recycle_depth)
        i = state["i"]
        state["i"] = i + 1
        if i + 1 >= steps:
            cap.sampler_done = True
        if i < steps:
            cap.save_blob(
                f"sampler_step{i}__{mode}",
                {"step": i, "mode": mode, "recycle_depth": recycle_depth,
                 "batch": _detach(batch), "output": _detach(out)},
                cls="SamplerStep", module="predict_for_sampling", call=i,
                extra={"mode": mode, "recycle_depth": recycle_depth},
            )
        return out

    model.predict_for_sampling = wrapped


def force_cpu_fp32_trainer() -> None:
    """Run upstream's generation Trainer on CPU in fp32.

    `run_generation` hardcodes `accelerator="gpu"` and takes its precision from the config,
    so on a box with no CUDA the Trainer raises "No supported gpu backend found!". The
    reference environment overrides those three arguments and nothing else: same code path,
    same model, same sampler.
    """
    import lightning as L

    base = L.Trainer

    class CPUTrainer(base):
        def __init__(self, *args, **kwargs):
            kwargs["accelerator"] = "cpu"
            kwargs["devices"] = 1
            kwargs["num_nodes"] = 1
            kwargs["precision"] = "32-true"
            super().__init__(*args, **kwargs)

    L.Trainer = CPUTrainer


def _caused_by(exc: BaseException, cls: type) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, cls):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True)
    ap.add_argument("--out", default="/home/moritz/rfd4-goldens")
    ap.add_argument("--records", default="/home/moritz/rfd4-goldens/records.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-module", type=int, default=1)
    ap.add_argument("--config-dir", default=None,
                    help="compose from this config directory instead of upstream's configs/ "
                         "(the antibody and nanobody examples ship their own config)")
    ap.add_argument("--config-name", default="inference")
    ap.add_argument("--stop-after-captures", action="store_true",
                    help="end generation as soon as every hooked module has its golden "
                         "(module goldens only, no finished design)")
    ap.add_argument("--sampler-steps", type=int, default=3,
                    help="record this many sampler steps (predict_for_sampling in/out)")
    ap.add_argument("--per-class", type=int, default=0,
                    help="hook at most N module instances per class, sampled evenly (0 = all)")
    ap.add_argument("--f64", action="store_true",
                    help="also run each hooked module on the same inputs in float64 and store that output")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--ckpt", default="nvidia/RFD4-Proteina")
    ap.add_argument("overrides", nargs="*", help="hydra overrides for rfd4.generate")
    a = ap.parse_args()

    sha = os.environ.get("RFD4_SHA", "2db07bb444d91695f8c10e6acb7648e9297f9221")
    torch.set_default_dtype(torch.float32)
    seed_everything(a.seed)

    import rfd4_proteina as rfd4

    model = rfd4.Proteina.from_pretrained(a.ckpt)
    cap = Capturer(Path(a.out), a.feature, a.seed, sha, a.classes.split(","),
                   a.per_module, Path(a.records), f64=a.f64,
                   stop_when_done=a.stop_after_captures)
    n = cap.attach(model, per_class=a.per_class)
    wrap_sampler(cap, model, a.sampler_steps)
    print(f"[capture] hooked {n} modules for feature {a.feature}", flush=True)
    seed_everything(a.seed)
    force_cpu_fp32_trainer()
    def run():
        if a.config_dir is None:
            rfd4.generate(model, overrides=list(a.overrides))
            return
        # Same path rfd4.generate takes, with the example's own config directory.
        from hydra import compose, initialize_config_dir
        from rfproteina.generate import run_generation

        with initialize_config_dir(config_dir=a.config_dir, version_base="1.3"):
            cfg = compose(config_name=a.config_name, overrides=list(a.overrides))
        run_generation(cfg, model=model)

    try:
        run()
    except CapturesComplete:
        print("[capture] every hooked module captured, generation stopped early", flush=True)
    except BaseException as exc:  # Lightning wraps a hook exception
        if not _caused_by(exc, CapturesComplete):
            raise
        print("[capture] every hooked module captured, generation stopped early", flush=True)
    cap.detach_all()
    print(f"[capture] wrote {sum(min(v, a.per_module) for v in cap.counts.values())} captures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

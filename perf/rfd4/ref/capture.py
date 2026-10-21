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


class Capturer:
    """Hooks selected submodules and writes <out>/<feature>/<slug>.pt per capture."""

    def __init__(self, out_dir: Path, feature: str, seed: int, sha: str,
                 classes: list[str], per_module: int, records: Path):
        self.dir = out_dir / feature
        self.dir.mkdir(parents=True, exist_ok=True)
        self.feature = feature
        self.seed = seed
        self.sha = sha
        self.classes = set(classes)
        self.per_module = per_module
        self.records = records
        self.counts: dict[str, int] = {}
        self.handles: list = []

    def attach(self, model: torch.nn.Module) -> int:
        n = 0
        for name, mod in model.named_modules():
            if type(mod).__name__ in self.classes:
                self.handles.append(mod.register_forward_hook(self._make_hook(name), with_kwargs=True))
                n += 1
        return n

    def detach_all(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

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
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with self.records.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

        return hook


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True)
    ap.add_argument("--out", default="/home/moritz/rfd4-goldens")
    ap.add_argument("--records", default="/home/moritz/rfd4-goldens/records.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-module", type=int, default=1)
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
                   a.per_module, Path(a.records))
    n = cap.attach(model)
    print(f"[capture] hooked {n} modules for feature {a.feature}", flush=True)
    seed_everything(a.seed)
    rfd4.generate(model, overrides=list(a.overrides))
    cap.detach_all()
    print(f"[capture] wrote {sum(min(v, a.per_module) for v in cap.counts.values())} captures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

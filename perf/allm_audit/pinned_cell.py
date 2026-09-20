#!/usr/bin/env python3
"""Run cell.py with today's HF revision pins installed, whatever tree it is folding out of.

Why this exists. `cell.py` is kept byte-identical to `pvx-baseline`'s copy (md5
21e0770f080a4d965203b59a193107be) so this row's arms, `pvx-didittransfer`'s and
`pvx-baseline`'s all time the same region. It therefore cannot carry a fix. And the old arms
need one: on 2026-09-14 19:17 UTC upstream re-published `biohub/ESMFold2`, `biohub/ESMFold2-Fast`
and `biohub/ESMC-6B` in place, in a schema the vendored loader cannot parse, and took the public
service down for six hours. `origin/main` answered with `weights.HF_REVISIONS`, pinning every
third-party repo to the commit its port was verified against. A tree from 2026-08-14..20 predates
that table, asks the hub for `main`, and dies on
`DiffusionStructureHeadConfig.__init__() got an unexpected keyword argument 'architectures'`.

So the old arm is not measurable against the hub as it stands today, and there are two ways to
make it measurable. Substituting a newer checkpoint would make the ratio a weights change wearing
an optimization's clothes. Pinning restores exactly the checkpoint the published cell was taken
against -- the pins ARE the pre-2026-09-14 commits -- and gives both arms byte-identical weights,
which is the only way the ratio is a property of the tree.

The pins are read out of the NEW tree's `weights.py` as TEXT and never imported, because this
process has an old `tt_bio` on its path and importing the new one would shadow the tree under
test. Every download funnels through `huggingface_hub.hf_hub_download` / `snapshot_download`, so
filling `revision` there covers `transformers.from_pretrained` too. A repo with no pin is left
alone.

On the new tree this shim is a no-op by construction: that tree passes the same revisions itself.
It is run on BOTH arms anyway, so the two arms differ in the tree and in nothing else. Which
repos it actually pinned at run time is COUNTED and written into the result JSON, never assumed.
"""
from __future__ import annotations

import ast
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_WEIGHTS = Path(os.environ["ALLM_PIN_SOURCE"])          # <new tree>/tt_bio/weights.py


def read_pins(src: Path) -> dict[str, str]:
    """HF_REVISIONS out of a weights.py by parsing it, not by importing it."""
    tree = ast.parse(src.read_text())
    consts: dict[str, str] = {}
    pins: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            consts[tgt.id] = node.value.value
        if isinstance(tgt, ast.Name) and tgt.id == "HF_REVISIONS" \
                and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    continue
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    pins[k.value] = v.value
                elif isinstance(k, ast.Name) and k.id in consts:
                    pins[consts[k.id]] = v.value
    return pins


PINS = read_pins(NEW_WEIGHTS)
FIRED: dict[str, int] = {}


def install() -> None:
    import huggingface_hub as H

    def wrap(mod, name):
        fn = getattr(mod, name, None)
        if fn is None:
            return

        def pinned(*a, **kw):
            repo = kw.get("repo_id") or (a[0] if a else None)
            if kw.get("revision") is None and repo in PINS:
                kw["revision"] = PINS[repo]
                FIRED[repo] = FIRED.get(repo, 0) + 1
            return fn(*a, **kw)

        setattr(mod, name, pinned)

    for name in ("hf_hub_download", "snapshot_download"):
        wrap(H, name)
        for sub in ("file_download", "_snapshot_download"):
            m = getattr(H, sub, None)
            if m is not None:
                wrap(m, name)
    # transformers binds the symbol at import time, so patch its copy as well.
    try:
        import transformers.utils.hub as TH
        wrap(TH, "hf_hub_download")
    except Exception:                  # noqa: BLE001 -- absent is fine, nothing to pin
        pass


install()

out = None
for i, arg in enumerate(sys.argv):
    if arg == "--out":
        out = Path(sys.argv[i + 1])

sys.argv = [str(HERE / "cell.py")] + sys.argv[1:]
try:
    runpy.run_path(str(HERE / "cell.py"), run_name="__main__")
finally:
    # The count, not the intent: a shim that pinned nothing would write {} here and that is
    # what tells the old arm apart from the new one.
    if out is not None and out.is_file():
        try:
            res = json.loads(out.read_text())
            res["hf_pins_available"] = PINS
            res["hf_pins_fired"] = dict(FIRED)
            out.write_text(json.dumps(res, indent=1))
        except Exception as e:         # noqa: BLE001
            print(f"note: could not record pins into {out}: {e!r}", flush=True)

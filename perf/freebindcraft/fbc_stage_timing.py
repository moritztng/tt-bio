"""Add per-stage timers to a FreeBindCraft checkout without touching a line number.

`--verbose` already times the four gradient stages of the hallucination loop, the OpenMM relax and
the open-source scoring. It does not time the MPNN sequence generation or the two AF2 validation
predictions, which is exactly the boundary the feasibility split turns on: those two are ordinary
inference and portable, the gradient stages are not.

Rather than patch by line number, this appends a guarded block to the bottom of the two modules
that define the stages. `functions/__init__.py` star-imports the modules, and each module is fully
executed before anything imports a name out of it, so rebinding at module bottom is enough for
both `bindcraft.py` and the intra-package imports to pick up the wrapped versions.

    python perf/freebindcraft/fbc_stage_timing.py --repo /work/FreeBindCraft
    FBC_TIMING_LOG=/work/out/stages.jsonl python -u bindcraft.py ... --verbose

Re-running is a no-op. With FBC_TIMING_LOG unset the shim does nothing, so a patched checkout still
behaves exactly like an unpatched one.
"""

import argparse
import pathlib

MARKER = "# --- tt-bio FreeBindCraft stage timing shim ---"

BLOCK = '''

{marker}
# Appended by perf/freebindcraft/fbc_stage_timing.py. Inert unless FBC_TIMING_LOG is set.
import os as _fbc_os, json as _fbc_json, time as _fbc_time, functools as _fbc_ft

_FBC_LOG = _fbc_os.environ.get("FBC_TIMING_LOG")
if _FBC_LOG:
    def _fbc_timed(_stage, _fn):
        @_fbc_ft.wraps(_fn)
        def _wrapper(*a, **kw):
            _t0 = _fbc_time.time()
            _ok = True
            try:
                return _fn(*a, **kw)
            except BaseException:
                _ok = False
                raise
            finally:
                with open(_FBC_LOG, "a") as _fh:
                    _fh.write(_fbc_json.dumps({{
                        "stage": _stage,
                        "s": round(_fbc_time.time() - _t0, 4),
                        "ok": _ok,
                        "t_end": round(_fbc_time.time(), 3),
                    }}) + "\\n")
        return _wrapper

{wraps}
'''

TARGETS = {
    "functions/colabdesign_utils.py": [
        "binder_hallucination",
        "mpnn_gen_sequence",
        "predict_binder_complex",
        "predict_binder_alone",
    ],
    "functions/pyrosetta_utils.py": [
        "pr_relax",
        "score_interface",
        "align_pdbs",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="FreeBindCraft checkout to patch in place")
    args = ap.parse_args()

    for rel, names in TARGETS.items():
        path = pathlib.Path(args.repo) / rel
        text = path.read_text()
        if MARKER in text:
            print(f"already patched: {rel}")
            continue
        missing = [n for n in names if f"def {n}(" not in text]
        if missing:
            raise SystemExit(f"{rel}: expected functions not found, upstream moved them: {missing}")
        wraps = "\n".join(f'    {n} = _fbc_timed("{n}", {n})' for n in names)
        path.write_text(text + BLOCK.format(marker=MARKER, wraps=wraps))
        print(f"patched {rel}: {', '.join(names)}")


if __name__ == "__main__":
    main()

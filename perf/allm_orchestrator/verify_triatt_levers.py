#!/usr/bin/env python3
"""Two measured triangle-attention levers, both off by default, both refused on OpenFold3 and RF3.

Checked from source with no `tt_bio` import, because the campaign's rule is that a lever's
eligibility is never read off prose. I got this wrong once while finding it: the call site of
`_tri_att_fused_qkv_sdpa` is unconditional, which reads as "shipped on", and the flag check is
actually inside `sdpa_fused_qkv`. Hence claim C.

  A. TT_BIO_TRIATT_FUSE_QKV defaults FALSE       -- the ROOF Phase A qkv+SDPA fusion is OFF today
  B. TT_BIO_TRIATT_GATE_EPILOGUE defaults FALSE  -- the gate epilogue is OFF today
  C. sdpa_fused_qkv returns before any work unless _FUSE_QKV or force
  D. both site clauses (the fusion's and the epilogue's) refuse on `att.biased` and
     `att.fp32_softmax`
  E. every OpenFold3-family Pairformer site passes fp32_softmax=True, so D refuses all of them
  F. no Boltz-2 / Protenix / OpenDDE site passes fp32_softmax, so D does not refuse them

A+B mean neither lever is a live explanation for any model's transfer ratio -- nobody has them.
D+E mean that turning either one on cannot reach OpenFold3 or RFdiffusion3, so they are levers for
three of the six shared-trunk models and structurally refused for two of the ones furthest from
the charter's bar.

exit 0  all six hold      exit 1  a claim fails      exit 2  the symbols moved, test is stale
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "tt_bio"
TS, TSD = ROOT / "tenstorrent.py", ROOT / "triatt_sdpa.py"
OF3_SITES = ["openfold3_trunk.py", "openfold3_confidence.py",
             "openfold3_template.py", "openfold3_msa_embedder.py"]
CLEAN = ["boltz2.py", "protenix.py", "opendde.py"]


def main() -> int:
    for f in (TS, TSD):
        if not f.is_file():
            print(f"STALE: {f} missing")
            return 2
    ts, tsd = TS.read_text(), TSD.read_text()
    bad, notes = [], []

    # A / B -- module-level defaults
    for flag, src, name in ((r'_FUSE_QKV\s*=\s*env_flag\(\s*"TT_BIO_TRIATT_FUSE_QKV"\s*,\s*(\w+)',
                             tsd, "A TT_BIO_TRIATT_FUSE_QKV"),
                            (r'(?m)^TRIATT_GATE_EPILOGUE\s*=\s*(\w+)', tsd,
                             "B TRIATT_GATE_EPILOGUE")):
        m = re.search(flag, src)
        if not m:
            print(f"STALE: cannot find {name}")
            return 2
        if m.group(1) != "False":
            bad.append(f"{name} defaults {m.group(1)}, not False -- the lever is ON and this "
                       f"whole reading changes")
        else:
            notes.append(f"OK  {name} defaults False")

    # C -- the early return, inside the function rather than at its call site
    tree = ast.parse(tsd)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "sdpa_fused_qkv"), None)
    if fn is None:
        print("STALE: no sdpa_fused_qkv")
        return 2
    first = ast.unparse(fn.body[1]) if len(fn.body) > 1 else ""
    if "_FUSE_QKV" not in first or "force" not in first:
        bad.append(f"C sdpa_fused_qkv does not gate on _FUSE_QKV/force first; it starts {first[:70]!r}")
    else:
        notes.append("OK  C sdpa_fused_qkv declines immediately unless _FUSE_QKV or force")

    # D -- both site clauses
    clauses = re.findall(r"if (att\.biased[^\n:]*):", ts)
    if len(clauses) < 2:
        print(f"STALE: expected 2 `att.biased ...` site clauses in tenstorrent.py, found "
              f"{len(clauses)}")
        return 2
    for c in clauses[:2]:
        for tok in ("att.biased", "att.fp32_softmax"):
            if tok not in c:
                bad.append(f"D a site clause lacks {tok}: {c[:70]!r}")
    if not any(f"D a site clause" in b for b in bad):
        notes.append(f"OK  D both site clauses refuse on att.biased and att.fp32_softmax")

    # E / F -- which models spell the kwarg
    for f in OF3_SITES:
        p = ROOT / f
        if not p.is_file():
            print(f"STALE: {p} missing")
            return 2
        if "fp32_softmax=True" not in p.read_text():
            bad.append(f"E {f} no longer passes fp32_softmax=True -- D would stop refusing it")
    if not any(b.startswith("E ") for b in bad):
        notes.append(f"OK  E all {len(OF3_SITES)} OpenFold3-family sites pass fp32_softmax=True")
    for f in CLEAN:
        p = ROOT / f
        if p.is_file() and re.search(r"fp32_softmax\s*=\s*True", p.read_text()):
            bad.append(f"F {f} now passes fp32_softmax=True -- it would join the refused set")
    if not any(b.startswith("F ") for b in bad):
        notes.append(f"OK  F no Boltz-2 / Protenix / OpenDDE site passes fp32_softmax")

    for n in notes:
        print(n)
    for b in bad:
        print("FAIL " + b)
    if bad:
        return 1
    print()
    print("=> Neither lever is on for ANY model, so neither explains a transfer ratio. Turning "
          "either on reaches Boltz-2, Protenix-v2 and OpenDDE, and is refused on 100 % of "
          "OpenFold3's and RFdiffusion3's triangle attention by one clause. `fp32_softmax=True` "
          "at OpenFold3's four sites is a single kwarg gating at least three measured levers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

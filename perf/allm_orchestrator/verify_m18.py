#!/usr/bin/env python3
"""Acceptance test for the campaign's lead candidate: is OpenFold3 plumbed for the fused SDPA?

The orchestrator handed `allm-gates` a four-line change and a pile of claims about the code it
touches. Claims handed to another row should be executable, so this checks them, and afterwards it
is the row's own acceptance test: it FAILS while OpenFold3's Pairformer sites are unplumbed and
PASSES once they carry `tri_att_sdpa_hifi`.

What it asserts, all by AST against the tree rather than by importing tt_bio (importing it is heavy
and this must run anywhere, including with no device):

  1. `Pairformer.__init__` really accepts `tri_att_sdpa_hifi`   -- otherwise the change cannot be
     made at all and the whole candidate is void.
  2. `triatt_sdpa_hifi_site` really exists and defaults False   -- plumbing it must be inert until
     someone flips it, which is what makes the change safe to land ahead of its A/B.
  3. `Boltz-2` passes it and `OpenFold3` does not               -- the divergence the candidate is
     about. If Boltz-2 ever stops passing it, the comparison this campaign rests on has moved.
  4. OpenFold3's four Pairformer sites are the ones named       -- so the row edits the right lines.

Exit 0 when OpenFold3 is plumbed at all four, 1 when it is not (the state this ships in), 2 if a
structural assumption is wrong, which is the case that invalidates the candidate rather than merely
leaving it undone.

    python3 perf/allm_orchestrator/verify_m18.py
"""
import ast
import sys
from pathlib import Path

TS = Path("tt_bio/tenstorrent.py")
OF3_SITES = {
    "openfold3.trunk": Path("tt_bio/openfold3_trunk.py"),
    "openfold3.template": Path("tt_bio/openfold3_template.py"),
    "openfold3.msa": Path("tt_bio/openfold3_msa_embedder.py"),
    "openfold3.confidence": Path("tt_bio/openfold3_confidence.py"),
}
KWARG = "tri_att_sdpa_hifi"
PAIRFORMER = {"Pairformer", "PairformerLayer", "PairformerModule"}


def _tree(p):
    return ast.parse(p.read_text(errors="replace"))


def _pairformer_calls(path):
    """-> [(lineno, {kwarg names})] for Pairformer-family constructions in `path`."""
    try:
        tree = _tree(path)
    except (OSError, SyntaxError):
        return []
    local = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.endswith("tenstorrent"):
            for a in n.names:
                local[a.asname or a.name] = a.name
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn, name = n.func, None
        if isinstance(fn, ast.Name):
            name = local.get(fn.id)
        elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) \
                and fn.value.id == "tenstorrent":
            name = fn.attr
        if name in PAIRFORMER:
            out.append((n.lineno, {k.arg for k in n.keywords if k.arg}))
    return out


def main():
    if not TS.is_file():
        print("run me from the repo root", file=sys.stderr)
        return 2
    ts = _tree(TS)
    hard = []

    # 1 + 2: the mechanism exists and is inert by default.
    sig = next((f for f in ast.walk(ts) if isinstance(f, ast.FunctionDef)
                and f.name == "__init__"
                and KWARG in [a.arg for a in f.args.args + f.args.kwonlyargs]), None)
    if sig is None:
        hard.append(f"no __init__ in tenstorrent.py accepts `{KWARG}` -- the candidate is void")

    site = next((f for f in ast.walk(ts) if isinstance(f, ast.FunctionDef)
                 and f.name == "triatt_sdpa_hifi_site"), None)
    if site is None:
        hard.append("triatt_sdpa_hifi_site is gone -- the per-site A/B grammar is void")
    else:
        dflt = site.args.defaults[-1] if site.args.defaults else None
        if not (isinstance(dflt, ast.Constant) and dflt.value is False):
            hard.append("triatt_sdpa_hifi_site no longer defaults False -- plumbing it would stop "
                        "being inert, so it could not be landed ahead of its A/B")

    # 3: the divergence the candidate is about.
    b2 = [(ln, kw) for ln, kw in _pairformer_calls(Path("tt_bio/boltz2.py")) if KWARG in kw]
    if not b2:
        hard.append("no Boltz-2 Pairformer site passes it any more -- the comparison this "
                    "campaign rests on has moved; re-read M17 before acting")

    # 4: OpenFold3's four sites, and whether each is plumbed.
    plumbed, unplumbed = [], []
    for token, path in OF3_SITES.items():
        calls = _pairformer_calls(path)
        if not calls:
            hard.append(f"{path} constructs no Pairformer -- the site list in M18 is wrong")
            continue
        for ln, kw in calls:
            (plumbed if KWARG in kw else unplumbed).append(f"{path}:{ln} ({token})")

    w = sys.stdout.write
    w(f"Pairformer accepts {KWARG}          : {'yes' if sig else 'NO'}\n")
    w(f"triatt_sdpa_hifi_site, default False : {'yes' if site and not hard else 'see below'}\n")
    w(f"Boltz-2 sites passing it             : {len(b2)} "
      f"({', '.join(f'boltz2.py:{ln}' for ln, _ in b2) or '-'})\n")
    w(f"OpenFold3 sites plumbed              : {len(plumbed)} of "
      f"{len(plumbed) + len(unplumbed)}\n")
    w("    (line numbers here are the CALL's first line, which is where ast puts it. The ledger\n"
      "     cites the kwarg line inside the same call -- e.g. openfold3_trunk.py 137 here against\n"
      "     139 there. Same site, different referent; neither is wrong.)\n")
    for s in plumbed:
        w(f"    plumbed   {s}\n")
    for s in unplumbed:
        w(f"    UNPLUMBED {s}\n")

    if hard:
        w("\nSTRUCTURAL ASSUMPTION BROKEN -- the candidate needs re-reading, not just doing:\n")
        for h in hard:
            w(f"  - {h}\n")
        return 2
    if unplumbed:
        w(f"\nNOT YET DONE: {len(unplumbed)} OpenFold3 site(s) still take the default. This is the\n"
          f"state the campaign shipped in; the change is to add "
          f"`{KWARG}=triatt_sdpa_hifi_site(\"<token>\")` to each.\n")
        return 1
    w("\nDONE: every OpenFold3 Pairformer site is plumbed. Now A/B per site with\n"
      "TT_BIO_TRIATT_SDPA_HIFI_AB, clear the accuracy bar, and FLIP THE DEFAULTS -- a lever\n"
      "merged default-off is not shipped (ledger M22).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

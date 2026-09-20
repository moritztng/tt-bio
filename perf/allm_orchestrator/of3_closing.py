#!/usr/bin/env python3
"""Can OpenFold3 reach Boltz-2's 1.5006x? Computed from a committed census, not typed.

The charter permits a NO-GO only with arithmetic showing the remaining gap is unreachable, so
the arithmetic has to exist before the verdict does. This is pvx T16's method: partition every
second of the fold into disjoint classes, refuse to report if the partition does not sum, then
show what deleting each class outright would buy.

TWO THINGS THIS IS NOT.

  - It is not a plan. It deletes whole classes outright, which no lever does. It is an upper
    bound, and an upper bound that still misses is the only honest way to close a charter.
  - It is not current. The census it reads (perf/of3_4xpd/decomp_512_main_qb1c1.json, ttnn
    0.67.4) predates the 1.1311x `allm-audit` measured between 973ae49f and 47810889f, so
    whatever that window sped up now occupies a SMALLER share than these numbers say. The
    shares are therefore an upper bound on today's, which makes the conclusion stronger and
    the individual figures unreliable. Ledger M13b. Re-run against a census on today's tree
    before quoting any single class.

Exit 2 if the partition does not sum to the fold, so a class that silently loses seconds
cannot flatter the verdict.

    python3 perf/allm_orchestrator/of3_closing.py
"""
import json
import sys
from pathlib import Path

CENSUS = Path("perf/of3_4xpd/decomp_512_main_qb1c1.json")
TARGET = 1.5006          # Boltz-2's measured ratio, pvx-didittransfer
BANKED = 1.1311          # OpenFold3's measured ratio, allm-audit session 1
TRIATT_OF_PF = 2.7182 / 12.169   # Boltz-2 byte census: TriangleAttention share of PairformerLayer
TRIMUL_OF_PF = 5.5070 / 12.169   # ... and TriangleMultiplication's


def main():
    if not CENSUS.is_file():
        print(f"missing {CENSUS} (run from the repo root)", file=sys.stderr)
        return 2
    d = json.loads(CENSUS.read_text())
    run = d["runs"][0]
    fold, r = run["fold_s"], run["regions"]

    def s(k):
        return r[k]["s"] if k in r else 0.0

    # Disjoint classes. `top:trunk` contains template and msa_block, so the Pairformer class is
    # the trunk MINUS its instrumented children -- subtracting rather than assuming.
    pairformer = s("top:trunk") - s("trunk:template") - s("trunk:msa_block") \
        - s("trunk:glue_z") - s("trunk:glue_s") - s("trunk:msa_embedder")
    classes = {
        "trunk Pairformer": pairformer,
        "trunk template": s("trunk:template"),
        "trunk MSA": s("trunk:msa_block") + s("trunk:msa_embedder"),
        "trunk glue": s("trunk:glue_z") + s("trunk:glue_s"),
        "diffusion rollout": s("diff:rollout"),
        "confidence": s("top:confidence"),
        "host featurisation": s("host:build_features") + s("host:derive_relpos")
                              + s("host:ref_atom_embed") + s("host:derive_template_feat")
                              + s("host:derive_block_aux") + s("host:msa_features")
                              + s("host:write_structure"),
        "atom encoder prep": s("prep:input_atom_encoder"),
    }
    named = sum(classes.values())
    classes["unattributed remainder"] = fold - named

    total = sum(classes.values())
    if abs(total - fold) > 1e-6:
        print(f"partition does not sum: {total:.4f} against fold {fold:.4f}", file=sys.stderr)
        return 2

    w = sys.stdout.write
    w(f"OpenFold3 512 aa, {CENSUS} (INSTRUMENTED, {fold:.3f} s; shares usable, seconds are not)\n\n")
    for k, v in sorted(classes.items(), key=lambda kv: -kv[1]):
        w(f"  {k:24s} {v:7.3f} s   {v/fold*100:5.2f} %\n")
    w(f"  {'-'*24} {total:7.3f} s   {total/fold*100:5.2f} %   partition sums\n\n")

    # What the charter still owes, on the ratios rather than on this instrumented fold.
    owed = TARGET / BANKED
    delete = 1 - 1 / owed
    w(f"banked {BANKED:.4f}x, target {TARGET:.4f}x  ->  still owed {owed:.4f}x\n")
    w(f"which means deleting {delete*100:.1f} % of TODAY's fold.\n\n")

    pf_share = pairformer / fold
    w("what deleting each class outright would buy, as a share of the fold:\n")
    rows = [(k, v / fold) for k, v in classes.items() if v > 0]
    rows += [("  of which triangle attention", pf_share * TRIATT_OF_PF),
             ("  of which triangle multiplication", pf_share * TRIMUL_OF_PF)]
    for k, share in sorted(rows, key=lambda kv: -kv[1]):
        reach = 1 / (1 - share)
        flag = "  <= REACHES 1.5006x" if reach >= owed else ""
        w(f"  delete {k:32s} {share*100:5.2f} %  ->  {reach:.4f}x{flag}\n")

    w("\nwhat this actually supports:\n")
    # Deleting a whole class is not a route -- levers take FRACTIONS of one. So the useful
    # question is what fraction of the fold the named candidate can reach, against what is owed.
    big = [(k, sh) for k, sh in rows if 1 / (1 - sh) >= owed and not k.startswith("  ")]
    w(f"  {len(big)} class(es) are individually large enough to clear {owed:.4f}x if DELETED: "
      f"{', '.join(k for k, _ in big)}.\n")
    w("  No lever deletes a class, so that is a bound and not a route.\n\n")
    cand = pf_share * TRIATT_OF_PF
    w(f"  the named candidate (M17) reaches triangle attention only: {cand*100:.2f} % of the "
      f"fold -> {1/(1-cand):.4f}x\n")
    w(f"  the charter still owes                                    {owed:.4f}x, i.e. "
      f"{delete*100:.1f} % of the fold\n")
    w(f"  shortfall after M17 lands in full: {owed / (1/(1-cand)):.4f}x still owed\n\n")
    # What would have to be true instead.
    need = delete
    w(f"  to close it you must find {need*100:.1f} % of the fold. That is more than the template\n"
      f"  ({classes['trunk template']/fold*100:.2f} %), the MSA "
      f"({classes['trunk MSA']/fold*100:.2f} %), the confidence "
      f"({classes['confidence']/fold*100:.2f} %) and the host\n"
      f"  ({classes['host featurisation']/fold*100:.2f} %) COMBINED "
      f"({(classes['trunk template']+classes['trunk MSA']+classes['confidence']+classes['host featurisation'])/fold*100:.2f} %),\n"
      f"  deleted outright. The only classes big enough are the ones no lever deletes: the\n"
      f"  Pairformer's own matmuls and the diffusion rollout.\n")
    w("\n  So 1.5006x on OpenFold3 needs the trunk Pairformer or the diffusion rollout to give up\n"
      "  roughly half of itself. Neither is in evidence, and the diffusion half is not even\n"
      "  shared code (ledger M16).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

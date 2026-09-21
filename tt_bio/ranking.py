"""One sample-ranking rule for the AF3-lineage models: OpenFold3, OpenBind, RF3,
Protenix-v1/v2, OpenDDE.

Every model here writes N diffusion samples and serves one of them as ``<stem>.cif``. Which
one it serves was decided in four different places, and on a single-chain target the four
disagreed:

    openfold3_fold.sample_ranking_score   0.8*pLDDT + 0.2*pTM + 0.5*disorder - 100*clash
    rf3.confidence.ranking_score          1.0*pTM - 100*clash
    worker._protenix_emit._score          pTM, or pLDDT when pTM is 0
    boltz2 / boltzgen confidence_score    0.8*pLDDT + 0.2*pTM

AF3 SI 5.9.3 is ``0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash``, and each of the first
three is that formula with a different improvisation for the case AF3 does not cover: a target
with no interface, where ipTM averages over cross-chain pairs that do not exist and is
identically zero. The improvisations were independent, so the family ranked the same monomer
three ways.

``of3t-confhead`` measured which of them is right, on 1UBQ over nine seeds and five samples:
pLDDT orders a model's own diffusion samples against true Ca-RMSD at Spearman +0.41 to +0.46,
pTM at +0.14 to +0.33, and the collapsed pTM rule selects a structure worse than picking at
random on a corrected trunk. So ipTM's 0.8 goes to pLDDT, which is per-residue, is already
computed at every one of these sites for the B-factor column, and is what Boltz-2 weights 4/5.

**With an interface this is AF3's formula, unchanged to the bit, at every site.** A site that
does not compute the disorder term or the clash indicator passes 0.0 for it and gets back
exactly the expression it had before, which is what makes the interface branch a no-op for
RF3 and Protenix rather than a change they have to be re-measured for.
"""
from __future__ import annotations

import json
import os

#: Set to a path to append one JSON line per call: the rule's inputs, its output, and the
#: call's index within the process. A ranking rule is post-forward, so re-ranking a recorded
#: fold offline under a different rule gives exactly the structure that rule would have
#: served -- which is how `perf/of3t_rankunify/` prices a rule change without re-folding.
_RECORD = os.environ.get("TT_BIO_RANK_RECORD") or None
_seq = 0


def ranking_score(*, ptm: float, iptm: float | None = None, plddt: float = 0.0,
                  disorder: float = 0.0, has_clash: float = 0.0) -> float:
    """Score one diffusion sample. Higher is served.

    ``iptm`` is None or 0.0 when the target has no interface to compute it from; that is the
    only case in which this differs from AF3 SI 5.9.3. ``disorder`` is the AF3 RASA term and
    ``has_clash`` the inter-chain polymer clash indicator; both default to 0.0 so a caller
    that does not compute them reproduces AF3's formula over the terms it does have.
    """
    ptm = float(ptm or 0.0)
    plddt = float(plddt or 0.0)
    interface = 0.8 * float(iptm) if iptm else 0.8 * plddt
    score = interface + 0.2 * ptm + 0.5 * float(disorder) - 100.0 * float(has_clash)
    if _RECORD:
        _emit(ptm=ptm, iptm=iptm, plddt=plddt, disorder=disorder,
              has_clash=has_clash, score=score)
    return score


def _emit(**row) -> None:
    global _seq
    row["i"] = _seq
    row["pid"] = os.getpid()
    _seq += 1
    with open(_RECORD, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")

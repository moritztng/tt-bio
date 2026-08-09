"""What alignment depth does the TT side actually fold at?

The GPU leg consumes unique(a3m rows) + 1, which is 35 for prot117 and 36 for prot300,
and gpu_bench asserts that against a hardcoded 35. Whether that is a real fairness
problem depends entirely on what the TT featurizer consumes for the same file, which
nobody has measured. Pure host work, so no card is needed.
"""
import sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "scripts" / "gpu_vs_tt"))
import tt_baseline
from tt_bio.main import _read_bio_chains, _read_bio_constraints, _resolve_a3m_text
from tt_bio.protenix_data import build_complex_features

for name, yaml, a3m in [
    ("prot117", HERE / "examples/prot.yaml",   HERE / "scripts/gpu_vs_tt/fixtures/prot117.a3m"),
    ("prot300", HERE / "examples/prot300.yaml", HERE / "scripts/gpu_vs_tt/fixtures/prot300.a3m"),
]:
    rows = a3m.read_text().split("\n")
    hdrs = [i for i, l in enumerate(rows) if l.startswith(">")]
    seqs = [rows[i + 1] for i in hdrs]
    with tempfile.TemporaryDirectory() as td:
        msa_dir = Path(td)
        reported = tt_baseline.seed_msa_cache(yaml, a3m, msa_dir)
        chains = _read_bio_chains(yaml)
        bonds = _read_bio_constraints(yaml)
        specs = [(cseq, _resolve_a3m_text(spec, cseq, msa_dir) if mt == "protein" else None, mt)
                 for _cid, cseq, spec, mt in chains]
        feats = build_complex_features(specs, chain_ids=[c for c, _s, _sp, _m in chains],
                                       bonds=bonds)
    msa = feats.get("msa")
    depth = tuple(msa.shape) if msa is not None else None
    print(f"{name}: a3m rows={len(seqs)} unique={len(set(seqs))} "
          f"| harness reports n_msa={reported} "
          f"| GPU would consume {len(set(seqs)) + 1} "
          f"| TT feats['msa'].shape={depth}")

"""Capture the WHOLE host featurizer output for one query, from either an upstream
openfold3 tree or from tt-bio, and diff two captures key by key.

Three trees matter here and they cannot share a process (each exposes a top-level
``openfold3``), so each capture is its own run:

    # upstream v0.5.0 (the OpenBind reference)
    PYTHONPATH=/home/ttuser/ob0_refdeps OF3_TREE=/home/ttuser/ob0_upstream \\
        OUT=/tmp/feat_v050.npz <refpy> scripts/ob0_featurizer_capture.py --query <q.json>

    # tt-bio's vendored pin, unpatched
    PYTHONPATH=/home/ttuser/ob0_refdeps OF3_TREE=/tmp/pin_of3 \\
        OUT=/tmp/feat_pin.npz <refpy> scripts/ob0_featurizer_capture.py --query <q.json>

    # tt-bio itself, either checkpoint flavour
    <devpy> scripts/ob0_featurizer_capture.py --query <q.json> --tt-bio --openbind \\
        --out /tmp/feat_ttbio_ob.npz

    # diff
    <anypy> scripts/ob0_featurizer_capture.py --compare /tmp/feat_v050.npz /tmp/feat_ttbio_ob.npz

The query JSON must pin ``main_msa_file_paths``, and the file's BASENAME must be a
``MSASettings.max_seq_counts`` key (``colabfold_main.a3m``) or the MSA is silently dropped
and the run dies in ``parse_msas`` with an IndexError.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def capture_upstream(tree: str, query_json: str) -> dict:
    sys.path.insert(0, tree)
    from openfold3.core.data.framework.single_datasets.inference import InferenceDataset
    from openfold3.core.config.pocket_sampling_config import PocketSamplingSettings
    from openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessorSettings,
    )
    from openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings, TemplateSettings,
    )
    from openfold3.projects.of3_all_atom.config.dataset_configs import InferenceJobConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    print(f"# tree: {sys.modules['openfold3'].__file__ if 'openfold3' in sys.modules else tree}")

    query_set = InferenceQuerySet.from_json(query_json)
    # The same inference defaults tt_bio.openfold3_data.build_openfold3_features uses.
    msa = MSASettings(subsample_main=False)
    if "cfdb_hits" not in msa.max_seq_counts:
        msa.max_seq_counts["cfdb_hits"] = 100000000
        msa.aln_order.insert(msa.aln_order.index("uniref90_hits") + 1, "cfdb_hits")
    cfg = InferenceJobConfig(
        query_set=query_set, seeds=[0], msa=msa,
        template=TemplateSettings(take_top_k=True),
        template_preprocessor_settings=TemplatePreprocessorSettings(mode="predict"),
        pocket_sampling=PocketSamplingSettings(),
    )
    ds = InferenceDataset(cfg)
    query = next(iter(query_set.queries.values()))
    np.random.seed(0)
    return ds.create_all_features(query)


def capture_tt_bio(query_json: str, openbind: bool) -> dict:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    import torch

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E501
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    query = next(iter(InferenceQuerySet.from_json(query_json).queries.values()))
    torch.manual_seed(0)
    np.random.seed(0)
    return build_openfold3_features(query, openbind=openbind)


def _to_numpy(v):
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return None


def save(features: dict, out: str) -> None:
    arrays, skipped = {}, []
    for k, v in features.items():
        a = _to_numpy(v)
        if a is None or a.dtype == object:
            skipped.append(f"{k}:{type(v).__name__}")
            continue
        arrays[k] = a
    np.savez_compressed(out, **arrays)
    print(f"wrote {out}: {len(arrays)} arrays"
          + (f"; skipped {len(skipped)} non-array ({', '.join(skipped)})" if skipped else ""))
    for k in sorted(arrays):
        a = arrays[k]
        print(f"  {k:<32} {str(a.dtype):<10} {str(a.shape):<20}")


def compare(a_path: str, b_path: str) -> int:
    A, B = np.load(a_path), np.load(b_path)
    ka, kb = set(A.files), set(B.files)
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    shared = sorted(ka & kb)
    exact, diff, shape_mismatch = [], [], []
    for k in shared:
        x, y = A[k], B[k]
        if x.shape != y.shape:
            shape_mismatch.append((k, x.shape, y.shape))
            continue
        if x.dtype.kind in "SU" or y.dtype.kind in "SU":
            (exact if (x == y).all() else diff).append((k, "str", 0.0))
            continue
        if np.array_equal(x, y):
            exact.append((k, str(x.dtype), 0.0))
        else:
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            diff.append((k, str(x.dtype), float(np.abs(xf - yf).max())))
    print(f"### {a_path}  vs  {b_path}")
    print(f"keys: {len(shared)} shared, {len(only_a)} only-A, {len(only_b)} only-B")
    print(f"BIT-EXACT: {len(exact)}/{len(shared)}")
    if only_a:
        print(f"only in A: {', '.join(only_a)}")
    if only_b:
        print(f"only in B: {', '.join(only_b)}")
    for k, sa, sb in shape_mismatch:
        print(f"SHAPE  {k:<32} {sa} vs {sb}")
    for k, dt, d in sorted(diff, key=lambda t: -t[2]):
        print(f"DIFF   {k:<32} {dt:<10} maxdiff={d:.6g}")
    return 0 if not diff and not shape_mismatch and not only_a and not only_b else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", help="inference query JSON with main_msa_file_paths pinned")
    ap.add_argument("--tt-bio", action="store_true",
                    help="capture from tt_bio.openfold3_data instead of a bare tree")
    ap.add_argument("--openbind", action="store_true",
                    help="with --tt-bio: select the OpenBind featurizer flags")
    ap.add_argument("--out", default=os.environ.get("OUT"))
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.query or not args.out:
        ap.error("--query and --out (or $OUT) are required unless --compare is given")
    if args.tt_bio:
        features = capture_tt_bio(args.query, args.openbind)
    else:
        tree = os.environ.get("OF3_TREE")
        if not tree:
            ap.error("set OF3_TREE to an openfold3 tree, or pass --tt-bio")
        features = capture_upstream(tree, args.query)
    save(features, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

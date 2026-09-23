"""Count every fused-SDPA call a real tt-bio run makes, at the op, and score it.

    python3 perf/mgx_sdpa/census.py --label esmc-6b-126 --out census.jsonl -- \
        embed seqs.fasta --model esmc-6b --out_dir out/

Everything after `--` is handed to `python -m tt_bio.main`. `tt-bio predict` folds in spawned
worker processes, so a counter read in the launcher is always zero; this writes a
`sitecustomize.py` onto PYTHONPATH instead, and every process of the run wraps
`ttnn.transformer.scaled_dot_product_attention` and dumps its own counts at exit.

One key per (caller file:line, B, H, Lq, Lk, d, mask shape or None, q_chunk, k_chunk, grid).
The first call of each key is also scored by effect, off the returned tensor:
  * `pcc_fp32`: the op's output against torch fp32 SDPA on the same q, k, v and mask;
  * `pcc_zero_mask` (unmasked calls only): the op's output against the same call with an
    all-zero additive mask, which changes nothing mathematically.
Scoring is skipped above --score-max elements of B*H*Lq*Lk so a 1536-token triangle attention
does not pull gigabytes to the host. A census run is for counting; do not take a digest from it.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HOOK = r'''
import os
if os.environ.get("SDPA_CENSUS_DIR"):
    import atexit, json, sys, traceback
    import ttnn, torch
    _orig = ttnn.transformer.scaled_dot_product_attention
    _rows = {}
    _max = int(os.environ.get("SDPA_CENSUS_SCORE_MAX", str(1 << 27)))

    def _site():
        for fr in reversed(traceback.extract_stack()[:-2]):
            if "/tt_bio/" in fr.filename and fr.name not in ("fused_sdpa", "<lambda>", "_sdpa_masked"):
                return f"{fr.filename.split('/tt_bio/')[-1]}:{fr.lineno}:{fr.name}"
        return "?"

    def _pcc(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            return None
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

    def _wrap(q, k, v, *a, **kw):
        mask = kw.get("attn_mask", a[0] if a else None)
        pc = kw.get("program_config")
        B, H, Lq, d = (int(s) for s in q.shape)
        Lk = int(k.shape[2])
        qc = getattr(pc, "q_chunk_size", None) if pc is not None else None
        kc = getattr(pc, "k_chunk_size", None) if pc is not None else None
        g = getattr(pc, "compute_with_storage_grid_size", None) if pc is not None else None
        key = json.dumps([_site(), B, H, Lq, Lk, d,
                          None if mask is None else [int(s) for s in mask.shape],
                          qc, kc, None if g is None else [g.x, g.y],
                          str(q.dtype), bool(kw.get("is_causal", False))])
        o = _orig(q, k, v, *a, **kw)
        r = _rows.get(key)
        if r is not None:
            r["calls"] += 1
            if r["calls"] % 256 == 0:
                _dump()
            return o
        r = _rows[key] = {"calls": 1}
        try:
            return _score(r, o, q, k, v, mask, B, H, Lq, Lk, d, kw)
        finally:
            _dump()  # a spawned worker can be terminated before atexit runs

    def _score(r, o, q, k, v, mask, B, H, Lq, Lk, d, kw):
        if B * H * Lq * Lk > _max:
            r["scored"] = "skipped: over score-max"
            return o
        try:
            qh, kh, vh = (ttnn.to_torch(t).float()[:B, :H] for t in (q, k, v))
            qh, kh, vh = qh[:, :, :Lq, :d], kh[:, :, :Lk, :d], vh[:, :, :Lk, :d]
            mh = None
            if mask is not None:
                mh = ttnn.to_torch(mask).float()
                mh = mh[:, :, :Lq, :Lk]
            scale = kw.get("scale")
            ref = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=mh, scale=scale)
            oh = ttnn.to_torch(o).float()[:B, :H, :Lq, :d]
            r["pcc_fp32"] = _pcc(oh, ref)
            if mask is None:
                z = ttnn.from_torch(torch.zeros(B, 1, Lq, Lk, dtype=torch.bfloat16),
                                    device=q.device(), layout=ttnn.TILE_LAYOUT)
                kz = dict(kw); kz["attn_mask"] = z
                oz = _orig(q, k, v, **kz)
                ozh = ttnn.to_torch(oz).float()[:B, :H, :Lq, :d]
                r["pcc_zero_mask"] = _pcc(oh, ozh)
                r["pcc_fp32_zero_mask"] = _pcc(ozh, ref)
                ttnn.deallocate(oz); ttnn.deallocate(z)
        except Exception as e:  # a failed score must not fail the run it is counting
            r["scored"] = f"error: {str(e).splitlines()[0][:160]}"
        return o

    def _dump():
        if _rows:
            p = os.path.join(os.environ["SDPA_CENSUS_DIR"], f"{os.getpid()}.json")
            with open(p + ".tmp", "w") as f:
                json.dump(_rows, f)
            os.replace(p + ".tmp", p)

    ttnn.transformer.scaled_dot_product_attention = _wrap
    atexit.register(_dump)
'''

FIELDS = ("site", "B", "H", "Lq", "Lk", "d", "mask", "q_chunk", "k_chunk", "grid", "dtype",
          "causal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-max", type=int, default=1 << 27)
    ap.add_argument("cli", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cli = a.cli[1:] if a.cli[:1] == ["--"] else a.cli
    with tempfile.TemporaryDirectory() as hook, tempfile.TemporaryDirectory() as dump:
        Path(hook, "sitecustomize.py").write_text(HOOK)
        env = dict(os.environ, SDPA_CENSUS_DIR=dump, SDPA_CENSUS_SCORE_MAX=str(a.score_max),
                   PYTHONPATH=os.pathsep.join([hook, str(REPO)] + [p for p in os.environ.get(
                       "PYTHONPATH", "").split(os.pathsep) if p]))
        rc = subprocess.call([sys.executable, "-m", "tt_bio.main"] + cli, cwd=REPO, env=env)
        merged = {}
        for p in glob.glob(os.path.join(dump, "*.json")):
            for key, r in json.load(open(p)).items():
                m = merged.setdefault(key, dict(r, calls=0))
                m["calls"] += r["calls"]
    rows = []
    for key, r in merged.items():
        rows.append(dict(zip(FIELDS, json.loads(key)), **r))
    rows.sort(key=lambda r: (r["site"], r["Lq"], r["B"]))
    with open(a.out, "a") as f:
        f.write(json.dumps({"label": a.label, "cli": cli, "rc": rc,
                            "calls": sum(r["calls"] for r in rows),
                            "unmasked": sum(r["calls"] for r in rows if r["mask"] is None),
                            "rows": rows}) + "\n")
    print(f"{a.label}: rc={rc} calls={sum(r['calls'] for r in rows)} "
          f"unmasked={sum(r['calls'] for r in rows if r['mask'] is None)} keys={len(rows)}")
    for r in rows:
        print("  ", {k: r.get(k) for k in ("site", "B", "H", "Lq", "d", "mask", "q_chunk",
                                            "k_chunk", "calls", "pcc_fp32", "pcc_zero_mask")})
    return rc


if __name__ == "__main__":
    sys.exit(main())

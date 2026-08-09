#!/usr/bin/env python3
"""One arm of the ms/fold A/B for the reconciled batched_matmul, through the same
scripts/gpu_vs_tt/tt_baseline.py harness the campaign absolutes come from: 298 aa, 10 recycling
steps, 200 sampling steps, 1 sample, seed 0, MSA cache pre-seeded.

--arm off sets TT_BIO_BATCHED_MATMUL=0 before tt_bio is imported, which makes the chooser decline
everywhere and restores today's plain ttnn.matmul at all ten sites. The env var rather than a
module rebind: the openfold3 modules import `batched_matmul` into their own namespace, so rebinding
`tenstorrent.batched_matmul` would score a silent no-op as a win (G1's lesson).

The CIF sha256 is reported so the arms can be compared for bit-exactness at the fold, not just at
the op.
"""
import argparse, hashlib, json, os, statistics, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=["off", "on"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    os.environ["TT_BIO_BATCHED_MATMUL"] = "0" if a.arm == "off" else "1"

    import tt_bio.tenstorrent as T
    assert T._BATCHED_MATMUL_ON == (a.arm == "on"), "the arm did not take"

    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="bmm-fold-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml",
        Path(B.FIXTURES) / "prot300.a3m")
    cold_s, cold_m = one_fold()
    times, plddt = [], None
    for _ in range(a.repeat):
        t, m = one_fold()
        times.append(t)
        plddt = m["plddt"]
    cifs = sorted(Path(meta["struct_dir"]).glob("*.cif"))
    sha = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16] if cifs else None
    out = dict(model=a.model, arm=a.arm, target="examples/prot300.yaml", n_aa=298,
               recycling_steps=10, sampling_steps=200, cold_s=round(cold_s, 3),
               warm_s=[round(t, 3) for t in times],
               median_ms=round(statistics.median(times) * 1e3, 1),
               min_ms=round(min(times) * 1e3, 1), plddt=plddt, cif_sha256_16=sha,
               msa=bool(cold_m.get("msa")), n_tokens=cold_m.get("n_tokens"),
               hardware=meta["hardware"], card_type=meta.get("card_type"),
               aiclk_mhz=meta.get("aiclk_mhz"), loadavg=os.getloadavg())
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

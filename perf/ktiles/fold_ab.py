#!/usr/bin/env python3
"""One arm of the ms/fold A/B for the batched-matmul program config, through the same
scripts/gpu_vs_tt/tt_baseline.py harness the campaign absolutes come from: 298 aa CDK2,
10 recycling steps, 200 sampling steps, 1 sample, seed 0, MSA cache pre-seeded.

--arm off rebinds batched_matmul back to a plain ttnn.matmul in both modules before the
model loads, which is byte-for-byte what the eight call sites did before this branch.
The CIF sha256 is reported so the two arms can be compared for bit-exactness.
"""
import argparse, hashlib, json, statistics, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--arm", required=True,
                    choices=["on", "off", "on-no-b8", "atom-only", "dit-only"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    if a.arm == "off":
        plain = lambda x, y, compute_kernel_config=None: ttnn.matmul(
            x, y, compute_kernel_config=compute_kernel_config)
        T.batched_matmul = plain
        P.batched_matmul = plain
    if a.arm in ("atom-only", "dit-only"):
        # Attribution across the eight applied sites. tenstorrent.py holds the four DiT ones,
        # protenix.py the four atom-window ones, so rebinding one module's name splits them.
        plain = lambda x, y, compute_kernel_config=None: ttnn.matmul(
            x, y, compute_kernel_config=compute_kernel_config)
        if a.arm == "atom-only":
            T.batched_matmul = plain
        else:
            P.batched_matmul = plain
    if a.arm == "on-no-b8":
        # Bisecting the opendde fold-parity break: decline the one applied class no op-level
        # torch.equal covers, B=8 Mt=19 Kt=19 Nt=2 (4 calls per fold).
        orig = T._batched_matmul_config
        T._batched_matmul_config = (
            lambda b, mt, kt, nt, eb: None if b == 8 else orig(b, mt, kt, nt, eb))

    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="ktiles-msa-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml",
        Path(B.FIXTURES) / "prot300.a3m")
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"
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
               n_tokens=cold_m.get("n_tokens"), hardware=meta["hardware"],
               card_type=meta.get("card_type"), aiclk_mhz=meta.get("aiclk_mhz"))
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    from tt_bio.tenstorrent import cleanup
    cleanup()


main()

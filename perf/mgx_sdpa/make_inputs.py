"""Single-chain inputs of an exact residue count, for the SDPA census.

    python3 perf/mgx_sdpa/make_inputs.py <out_dir> 126 254 510 1534 ...

Writes `prot_<N>.yaml` (predict) and `seq_<N>.fasta` (embed, saprot): the CDK2 chain of
perf/size512/fixtures/cdk2x2_298.yaml tiled to N. The lengths that matter are the ones whose
token axis is ALREADY a multiple of 32 once the model adds its own tokens (ESMC and ESMFold2's LM
add BOS and EOS, so N = 32k - 2), because those are the calls that reached ttnn's SDPA unmasked.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CDK2 = re.search(r"sequence:\s*(\S+)",
                 (ROOT / "perf/size512/fixtures/cdk2x2_298.yaml").read_text()).group(1)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for n in map(int, sys.argv[2:]):
        seq = (CDK2 * (n // len(CDK2) + 1))[:n]
        (out / f"prot_{n}.yaml").write_text(
            f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n")
        (out / f"seq_{n}.fasta").write_text(f">cdk2_{n}\n{seq}\n")


if __name__ == "__main__":  # trace_probe imports CDK2
    main()

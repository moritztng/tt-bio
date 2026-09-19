#!/usr/bin/env python3
"""Host cost of the recycle-invariant MSA one-hot that Boltz-2's trunk rebuilds every cycle.

`tt_bio/tenstorrent.py` `MSAModule.forward` builds `m` from feats["msa"], has_deletion,
deletion_value and msa_paired, pads it, and uploads it with `_from_torch` (bf16). Every input is a
featurisation output, and `tt_bio/boltz2.py:5849` calls `msa_module(z, s_inputs, feats)` inside the
recycle loop, so `m` is identical on all four cycles and three of the four builds and uploads are
redundant. Anthropic's kits call this class `ec`/`fc` (esmfold2/CHANGES.md) and `hostfeat`
(openfold3/CHANGES.md).

CPU only -- this measures the host build, not the fold.
"""
import time

import torch

N_MSA, SEQ = 1024, 512  # the 512 aa cell: MSA bucketed to 1024 rows, 512 tokens
CYCLES = 4  # recycling_steps=3 -> 4 trunk cycles; matches the 16 MSALayer calls / 4 blocks


def main():
    torch.manual_seed(0)
    msa = torch.randint(0, 33, (1, N_MSA, SEQ), dtype=torch.long)
    rest = [torch.rand(1, N_MSA, SEQ) for _ in range(3)]

    def build():
        return torch.cat(
            [torch.nn.functional.one_hot(msa, num_classes=33).to(rest[0].dtype)]
            + [t.unsqueeze(-1) for t in rest], dim=-1)

    m = build()
    mib = m.numel() * 2 / 2 ** 20
    print(f"m {tuple(m.shape)} {m.dtype}: {m.numel() * m.element_size() / 2**20:.1f} MiB host, "
          f"{mib:.1f} MiB uploaded as bf16")

    def timeit(fn, n=9):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[n // 2], ts[0], ts[-1]

    med, lo, hi = timeit(build)
    cmed, _, _ = timeit(lambda: m.to(torch.bfloat16))
    print(f"host build : median {med*1e3:.1f} ms (min {lo*1e3:.1f}, max {hi*1e3:.1f}), n=9")
    print(f"bf16 cast  : median {cmed*1e3:.1f} ms")
    print(f"redundant  : {CYCLES-1} of {CYCLES} cycles = "
          f"{(med+cmed)*(CYCLES-1)*1e3:.0f} ms host + {mib*(CYCLES-1)/1024:.2f} GiB of upload")


if __name__ == "__main__":
    main()

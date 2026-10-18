"""Which step of AtomPairDevice's on-card z_to_p table build goes wrong (approbe: the z_to_p gather is the
first nondeterministic op; the two host-table gathers are bit-identical).

Builds the [P*P+1, 32] row-major table from a device-linear z_to_p three ways, REPS times each, and checks
the table and the gathered rows against host truth (to_torch of z_to_p before anything touches it):
  cur        pad on TILE, deallocate(z_to_p), to_layout, reshape, concat  (2a42df596)
  keep       same, without deallocate(z_to_p)
  rm_first   to_layout to ROW_MAJOR first, pad there, reshape, concat
Also prints whether pad's output shares z_to_p's buffer.

    python perf/b2_atomp/table_probe.py --out table.json [--reps 4]
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, ".")
import tt_bio.tenstorrent as T  # noqa: E402
import ttnn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=4)
args = ap.parse_args()

torch.manual_seed(0)
P, AZ, TZ, ROWS, H = 512, 16, 128, 4128, 128
dev = T.get_device()
z_tt = ttnn.from_torch(torch.randn(1, P, P, TZ) * 0.5, layout=ttnn.TILE_LAYOUT, device=dev,
                       dtype=ttnn.bfloat16)
wz_tt = ttnn.from_torch(torch.randn(TZ, AZ) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
index = torch.randint(0, P * P + 1, (ROWS, H))
index[:, -5:] = P * P


def addr(t):
    try:
        return t.buffer_address()
    except Exception as e:  # noqa: BLE001
        return repr(e)


def build(variant, z_to_p, info):
    if variant == "rm_first":
        rm = ttnn.to_layout(z_to_p, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(z_to_p)
        zp = ttnn.pad(rm, [(0, 0), (0, 0), (0, 0), (0, 32 - AZ)], 0.0)
        info["pad_aliases_input"] = addr(zp) == addr(rm)
        ttnn.deallocate(rm)
        table = ttnn.reshape(zp, (P * P, 32))
    else:
        zp = ttnn.pad(z_to_p, [(0, 0), (0, 0), (0, 0), (0, 32 - AZ)], 0.0)
        info["pad_aliases_input"] = addr(zp) == addr(z_to_p)
        if variant == "cur":
            ttnn.deallocate(z_to_p)
        table = ttnn.reshape(ttnn.to_layout(zp, ttnn.ROW_MAJOR_LAYOUT), (P * P, 32))
        ttnn.deallocate(zp)
    zero = ttnn.from_torch(torch.zeros(1, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16)
    table = ttnn.concat([table, zero], dim=0)
    return table


report = {}
for variant in ("keep", "rm_first", "cur"):
    rows = []
    for r in range(args.reps):
        z_to_p = ttnn.linear(z_tt, wz_tt)
        truth = torch.cat([ttnn.to_torch(z_to_p).float().reshape(P * P, AZ), torch.zeros(1, AZ)])
        info = {}
        table = build(variant, z_to_p, info)
        tab = ttnn.to_torch(table).float()
        idx = ttnn.from_torch(index.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                              dtype=ttnn.uint32)
        g = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        g = ttnn.to_torch(g).float().reshape(ROWS, H, -1)[..., :AZ]
        ttnn.deallocate(table)
        info.update(rep=r, table_shape=list(tab.shape),
                    table_max_err=float((tab[:, :AZ] - truth).abs().max()),
                    table_pad_absmax=float(tab[:, AZ:].abs().max()),
                    gather_max_err=float((g - truth[index]).abs().max()),
                    gather_digest=float(g.double().sum()))
        print(variant, json.dumps(info), flush=True)
        rows.append(info)
    report[variant] = rows
json.dump(report, open(args.out, "w"), indent=1)

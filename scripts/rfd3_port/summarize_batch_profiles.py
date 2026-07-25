"""Compare synchronized per-operation RFD3 profiles produced by profile_batch_forward."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_one", type=Path)
    parser.add_argument("batch_many", type=Path)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    one = json.loads(args.batch_one.read_text())
    many = json.loads(args.batch_many.read_text())
    rows_one = one["operations"]
    rows_many = many["operations"]
    print(
        f"operations: B={one['batch']} {len(rows_one)}, "
        f"B={many['batch']} {len(rows_many)}"
    )

    totals = defaultdict(lambda: [0, 0, 0])
    for row in rows_one:
        values = totals[row["operation"]]
        values[0] += 1
        values[1] += row["elapsed_ns"]
    for row in rows_many:
        totals[row["operation"]][2] += row["elapsed_ns"]
    print("\nAggregate synchronized time by operation:")
    for operation, (count, ns_one, ns_many) in sorted(
        totals.items(), key=lambda item: item[1][2] - item[1][1], reverse=True
    ):
        ratio = ns_many / ns_one if ns_one else float("nan")
        print(
            f"{operation:42s} n={count:4d} "
            f"B1={ns_one / 1e6:8.2f}ms B{many['batch']}={ns_many / 1e6:8.2f}ms "
            f"ratio={ratio:5.2f}x"
        )

    if len(rows_one) != len(rows_many):
        return
    paired = []
    for index, (row_one, row_many) in enumerate(zip(rows_one, rows_many)):
        if row_one["operation"] != row_many["operation"]:
            continue
        paired.append(
            (
                row_many["elapsed_ns"] - row_one["elapsed_ns"],
                index,
                row_one,
                row_many,
            )
        )
    print(f"\nTop {args.top} individual batch-scaling operations:")
    for delta, index, row_one, row_many in sorted(paired, reverse=True)[: args.top]:
        ratio = row_many["elapsed_ns"] / row_one["elapsed_ns"]
        print(
            f"#{index:4d} {row_one['operation']:32s} "
            f"{row_one['elapsed_ns'] / 1e6:7.3f}->{row_many['elapsed_ns'] / 1e6:7.3f}ms "
            f"{ratio:5.2f}x shapes={row_one['input_shapes']} -> {row_many['input_shapes']}"
        )


if __name__ == "__main__":
    main()

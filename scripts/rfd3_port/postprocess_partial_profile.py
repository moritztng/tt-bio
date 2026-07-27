"""Post-process a tracy capture whose device log covers only part of the run.

The on-device profiler buffer holds a bounded number of program records, so a
capture of a full RFD3 diffusion step (~6900 ops) drops the tail and tracy's
own post-processing aborts on an exact-count assertion.  The records that DID
land are a correct *prefix* of the run -- the buffer fills in issue order --
so host and device op lists still align positionally; only the tail is
missing.  This driver relaxes the count assertion so the prefix can be
summarised, and prints the coverage so every number derived from it is read
with the right denominator.

Usage:
    python postprocess_partial_profile.py <tracy-output-folder>
"""

from __future__ import annotations

import sys
from pathlib import Path


ASSERT_A = """            if device_op_id_debug and host_op_id_debug:
                assert False, ("""
REPLACE_A = """            logger.warning(
                f"PARTIAL DEVICE CAPTURE: {len(device_ops_time)} of "
                f"{len(host_ops_by_device[device])} ops on device {device}"
            )
            host_ops_by_device[device] = host_ops_by_device[device][: len(device_ops_time)]
            if False:
                assert False, ("""

ASSERT_B = """            else:
                assert (
                    False
                ), f"Device data mismatch: Expected {len(host_ops_by_device[device])} but received {len(device_ops_time)} ops on device {device}\""""
REPLACE_B = """            else:
                pass"""


def main() -> None:
    out_folder = Path(sys.argv[1])
    from tracy import process_ops_logs

    path = Path(process_ops_logs.__file__)
    source = path.read_text()
    if ASSERT_A not in source or ASSERT_B not in source:
        raise SystemExit("tracy source shape changed; update the patch")
    patched = source.replace(ASSERT_B, REPLACE_B, 1).replace(ASSERT_A, REPLACE_A, 1)

    namespace: dict = {"__name__": "tracy.process_ops_logs", "__file__": str(path)}
    exec(compile(patched, str(path), "exec"), namespace)

    namespace["process_ops"](
        output_folder=str(out_folder),
        name_append="",
        date=False,
        device_only=False,
        analyze_noc_traces=False,
        device_analysis_types=(),
        force_legacy_device_logs=False,
    )


if __name__ == "__main__":
    main()

"""Rewrite the 11x10 record of af2ig-trunk-device.json from this pass's shipped-arm report.

    python3 perf/bcx_land/make_record.py
"""
import json
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
OUT = WT / "perf/bcx_land/rerecord"
REC = WT / "docs/implementation-parity-data/af2ig-trunk-device.json"
sys.path.insert(0, str(WT / "scripts/af2_port"))
import device_floor as df  # noqa: E402


def report(name):
    text = (OUT / f"{name}.stdout").read_text()
    return json.loads(text[text.index("{"):])


def sysfs(field):
    return Path(f"/sys/class/tenstorrent/tenstorrent!0/{field}").read_text().strip()


def summary(rep):
    return "%d failing taps, pcc_min %r, failing scalars %s" % (
        rep["taps_failed"], rep["pcc_min"], ", ".join(sorted(df._failing_scalars(rep))) or "none")


shipped, again = report("shipped"), report("shipped_r2")
committed = json.loads(REC.read_text())
old = next(r for r in committed["records"] if r["compute_grid"] == shipped["compute_grid"])
assert [r["pcc"] for r in again["rows"]] == [r["pcc"] for r in shipped["rows"]], "no repro"

commit = subprocess.check_output(["git", "-C", str(WT), "rev-parse", "origin/main"], text=True).strip()
rec = dict(shipped)
rec["provenance"] = {
    "host": "qb1", "card": 3, "card_type": sysfs("tt_card_type"),
    "fw_bundle_ver": sysfs("tt_fw_bundle_ver"), "card_serial": sysfs("tt_serial"),
    "tt_kmd": Path("/sys/module/tenstorrent/version").read_text().strip(),
    "compute_grid": shipped["compute_grid"], "commit": commit, "date": "2026-09-24",
    "cause": ("638187138, c14's OuterProductMean output-stage layout, default-on for every model. "
              "It moves af2ig 0.150 -> 0.210 A CA RMSD from the JAX structure on the final "
              "recycle, inside the 0.60 A bar with a 1.84 A seed floor, and buys 14.3923 -> "
              "14.2346 s (1.0111x) on the 512 aa fold against a 0.0339 s A/A floor. "
              "TT_BIO_OPM_LEGACY_LAYOUT=1 on this commit reproduces the superseded record's "
              "device output bit for bit, so the lever is the whole move."),
    "reproduced": "a second process on the same card: identical pcc at all %d taps and identical "
                  "scalars" % len(shipped["rows"]),
    "reproduce_with": old["provenance"]["reproduce_with"],
    "controls": {
        "negative": "--template-host at the same grid: %s, FAIL against this record (%s)" % (
            summary(report("template_host")),
            df.af2ig_device_floor_verdict(report("template_host"), {"records": [rec]})[1]),
        "mutations": "--mutate extra-const: %s; --mutate block-order: %s; both FAIL" % (
            summary(report("mutate_extra_const")), summary(report("mutate_block_order"))),
    },
}
rec["floor"] = old["floor"]
committed["records"] = [rec if r is old else r for r in committed["records"]]
prev = committed.pop("superseded")
committed["superseded"] = [
    {"pcc_min": old["pcc_min"], "taps_failed": old["taps_failed"],
     "commit": old["provenance"]["commit"][:9], "date": old["provenance"]["date"],
     "stack": "qb1 p150a, card firmware 19.15.0.0, 11x10 Tensix grid",
     "why": "Recorded before 638187138 put the OuterProductMean output stage in a new layout on "
            "by default. See records[].provenance.cause."},
    *(prev if isinstance(prev, list) else [prev]),
]
REC.write_text(json.dumps(committed, indent=1, default=float) + "\n")
for name in ("shipped", "shipped_r2", "template_host", "mutate_extra_const", "mutate_block_order"):
    v = df.af2ig_device_floor_verdict(report(name), committed)
    print(f"{name}: {summary(report(name))} -> {v[0]}: {v[1][:160]}")

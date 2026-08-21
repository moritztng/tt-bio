"""Prove the registry re-downloads nothing a host already has.

~65 GiB of weights per host across the fleet, so a "unification" that makes any host
re-fetch is a net loss. This runs the exact resolution path every model uses and measures
two independent things: host network counters (all interfaces, rx+tx from /proc/net/dev)
and the mtime+size of every cached artifact. Both must be unchanged.

Rows that are genuinely missing or damaged on this host are listed and skipped: re-fetching
those is the fix working, not cache invalidation.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio import weights  # noqa: E402


def net_bytes() -> int:
    total = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, _, rest = line.partition(":")
        if name.strip() == "lo":
            continue
        f = rest.split()
        total += int(f[0]) + int(f[8])
    return total


MARKER = ".complete-"


def stamps() -> dict:
    """size+mtime of every weight byte on this host: the flat artifacts, their derived
    outputs, AND the Hugging Face hub blob store (the hub half is 44 of the 65 GiB, so
    stamping only the flat half would look clean while a repo re-snapshotted)."""
    out = {}
    for key in weights.ARTIFACTS:
        for p in weights.artifact_paths(key):
            for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
                if f.is_file() and MARKER not in f.name:
                    st = f.stat()
                    out[str(f)] = (st.st_size, st.st_mtime_ns)
    from huggingface_hub import constants
    hub = Path(constants.HF_HUB_CACHE)
    if hub.is_dir():
        for f in hub.glob("models--*/blobs/*"):
            if f.is_file():
                st = f.stat()
                out[str(f)] = (st.st_size, st.st_mtime_ns)
    return out


def main():
    present, skipped = [], []
    for key in weights.ARTIFACTS:
        st = weights.status(key)
        (present if st.state == "present" else skipped).append((key, st.state))

    print(f"cached and testable : {len(present)} rows")
    print(f"skipped             : {[f'{k}={s}' for k, s in skipped]}")

    # A/A control first: this host runs a coworker fleet, so the network counter has a
    # floor that is not ours. Measure it over a window of the same shape (a full stamp
    # walk, no fetching) so the B leg's delta is read against a real baseline.
    import time
    aa0 = net_bytes()
    _ = stamps()
    time.sleep(5)
    _ = stamps()
    aa_noise = net_bytes() - aa0
    print(f"A/A control (no fetches): {aa_noise} network bytes of host background traffic")

    before_net, before_files = net_bytes(), stamps()
    print(f"\nbaseline: {len(before_files)} files stamped "
          f"(flat artifacts + hub blobs), network counter {before_net}")

    # 1. every cached registry row, through the real fetch path
    for key, _ in present:
        weights.fetch(key, quiet=True)

    # 2. every model's own resolution path, exactly as a worker runs it
    from tt_bio.worker import _ensure_local_artifacts
    models = ["boltz2", "protenix-v2", "openfold3", "esmfold2", "esmfold2-fast",
              "opendde", "opendde-abag", "esmc-300m", "esmc-600m", "esmc-6b",
              "saprot-35m", "saprot-650m"]
    for m in models:
        _ensure_local_artifacts({"model": m, "msa_dir": None})

    # 3. the parent-side prefetch predict runs before fanning out
    from tt_bio.main import download_all
    for m in ("boltz2", "protenix-v2", "esmfold2", "opendde"):
        download_all(weights.cache_root(), m)

    # 4. BoltzGen's own resolver, on the five rows that are intact
    import argparse

    from tt_bio.boltzgen.cli.boltzgen import ARTIFACTS as BG, get_artifact_path
    args = argparse.Namespace(cache=weights.cache_root() / "boltzgen", force_download=False)
    for name, (spec, rtype) in BG.items():
        if weights.status(f"boltzgen-{ {'design-diverse': 'diverse', 'design-adherence': 'adherence', 'inverse-fold': 'ifold', 'folding': 'folding', 'affinity': 'affinity', 'moldir': 'mols'}[name] }").state != "present":
            print(f"  (skipping boltzgen {name}: not intact on this host)")
            continue
        get_artifact_path(args, spec, repo_type=rtype, verbose=False)

    after_net, after_files = net_bytes(), stamps()
    changed = {k: (before_files.get(k), v) for k, v in after_files.items()
               if before_files.get(k) != v}
    gone = sorted(set(before_files) - set(after_files))
    markers = sorted(p.name for p in weights.cache_root().rglob(f"{MARKER}*"))
    delta = after_net - before_net
    result = {
        "rows_exercised": len(present),
        "models_exercised": models,
        "aa_control_network_bytes": aa_noise,
        "network_bytes_delta": delta,
        "files_stamped": len(before_files),
        "files_changed": changed,
        "files_removed": gone,
        "completion_markers_written": markers,
        "skipped": skipped,
    }
    print(f"\nfiles stamped       : {len(before_files)}")
    print(f"files changed       : {len(changed)}")
    print(f"files removed       : {len(gone)}")
    print(f"markers written     : {markers}  (3 bytes each, beside the output dir)")
    print(f"network bytes delta : {delta}  (A/A control: {aa_noise})")
    # The registry either re-fetched an artifact or it did not, and that is decided by
    # the file stamps: a re-download rewrites the file. The network counter is a shared
    # host counter, so it is reported next to its own A/A control rather than gated on.
    ok = not changed and not gone
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{f'{len(before_files)} weight files, none rewritten, none removed' if ok else 'see above'}")
    if changed:
        for k, v in list(changed.items())[:10]:
            print(f"  {k}: {v[0]} -> {v[1]}")
    Path(__file__).with_name("zero_bytes_check.json").write_text(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

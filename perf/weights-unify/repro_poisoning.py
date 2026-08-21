"""Reproduce each cache-poisoning site against real artifacts, old code vs new.

Every case truncates a COPY of an artifact already on this host (no downloads) and
shows that the pre-registry gate accepts the wreckage while tt_bio.weights rejects it.
Run: python3 perf/weights-unify/repro_poisoning.py
"""
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio import weights  # noqa: E402

REAL = weights.cache_root()
OUT = []


def truncate_copy(src: Path, dst: Path, frac: float) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = int(src.stat().st_size * frac)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 20)
        fo.truncate(n)
    return n


def record(site, mechanism, old, new):
    OUT.append(dict(site=site, mechanism=mechanism, old_code=old, new_code=new))
    print(f"\n[{site}] {mechanism}")
    print(f"  old gate : {old}")
    print(f"  new gate : {new}")


def case_flat_file(scratch: Path):
    """hf_artifact / _download_file: `if not dest.exists(): download`."""
    scratch.mkdir(parents=True, exist_ok=True)
    src = REAL / "boltzgen" / "boltzgen1_ifold.ckpt"
    dst = scratch / "boltzgen1_ifold.ckpt"
    n = truncate_copy(src, dst, 0.60)
    old = f"dest.exists() -> {dst.exists()}, so it skips the download and returns {dst.name}"
    try:
        torch.load(dst, map_location="cpu", weights_only=False)
        old += "; torch.load unexpectedly SUCCEEDED"
    except Exception as e:
        old += f"; torch.load raises {type(e).__name__}: {str(e)[:70]}"
    new = (f"artifact_intact -> {weights.artifact_intact(dst)}, so fetch_hf_file "
           f"re-downloads ({n} of {src.stat().st_size} bytes present)")
    record("flat checkpoint (hf_artifact, _download_file)",
           "truncated file at the final path passes .exists()", old, new)


def case_url_download(scratch: Path):
    """_download_file wrote straight to dest with curl -C-/wget -c/aria2c: an
    interrupt leaves the truncated file AT the final path. Reproduced against a
    local HTTP server so no bytes leave the machine."""
    scratch.mkdir(parents=True, exist_ok=True)
    import http.server
    import socketserver
    import threading
    import urllib.request

    src = REAL / "boltzgen" / "boltzgen1_ifold.ckpt"
    serve_dir = scratch / "srv"
    serve_dir.mkdir(parents=True)
    shutil.copy2(src, serve_dir / "ckpt.pt")
    total = (serve_dir / "ckpt.pt").stat().st_size

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

        def log_message(self, *a):
            pass

    class Quiet(socketserver.TCPServer):
        # We deliberately drop the connection partway through; the reset is the point.
        def handle_error(self, *a):
            pass

    with Quiet(("127.0.0.1", 0), Handler) as srv:
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/ckpt.pt"

        # Old shape: the tool writes to the final path. Simulate the SIGKILL by
        # stopping the read partway, exactly where curl -o <dest> would leave it.
        dest = scratch / "old" / "ckpt.pt"
        dest.parent.mkdir(parents=True)
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as fo:
            fo.write(resp.read(total // 3))
        old = (f"killed at {dest.stat().st_size} of {total} bytes; the truncated file sits "
               f"at the final path and .exists() -> True, so every later run reuses it")

        # New shape: same interrupt, but staging is .<name>.part and the final path
        # is only written after verification.
        good = scratch / "new" / "ckpt.pt"
        good.parent.mkdir(parents=True)
        part = good.with_name(f".{good.name}.part")
        with urllib.request.urlopen(url) as resp, open(part, "wb") as fo:
            fo.write(resp.read(total // 3))
        interrupted = (good.exists(), part.stat().st_size)
        final = weights.fetch_url(url, good, quiet=True)
        new = (f"same interrupt leaves {interrupted[1]} bytes in {part.name}; final path "
               f"exists -> {interrupted[0]}. Next run resumes and lands "
               f"{final.stat().st_size} of {total} bytes, intact -> "
               f"{weights.artifact_intact(final)}")
        srv.shutdown()
    record("IPD checkpoint download (_download_file, rf3/rfd3)",
           "curl -C- / wget -c / aria2c wrote directly to the final path", old, new)


def case_tar_extract(scratch: Path):
    """download_mols gated on (cache/'mols').exists(): a dir left by a killed
    extractall passes. Built from a small synthetic tar so the check is exact."""
    scratch.mkdir(parents=True, exist_ok=True)
    tar_path = scratch / "lib.tar"
    src = scratch / "src" / "lib"
    src.mkdir(parents=True)
    for i in range(40):
        (src / f"m{i:03d}.pkl").write_bytes(b"x" * 64)
    with tarfile.open(tar_path, "w") as t:
        t.add(src, arcname="lib")

    # Old shape: extractall interrupted after 12 of 40 members.
    old_root = scratch / "old"
    (old_root / "lib").mkdir(parents=True)
    with tarfile.open(tar_path) as t:
        for i, m in enumerate(t):
            if i > 12:
                break
            t.extract(m, old_root)
    n_old = len(list((old_root / "lib").iterdir()))
    old = (f"(cache/'lib').exists() -> True with only {n_old} of 40 molecules, so the "
           f"library is never rebuilt and every lookup of a missing molecule fails")

    spec = weights.Derived("lib", "tar", min_entries=40)
    new_root = scratch / "new"
    (new_root / "lib").mkdir(parents=True)
    with tarfile.open(tar_path) as t:
        for i, m in enumerate(t):
            if i > 12:
                break
            t.extract(m, new_root)
    partial_ok = weights._derived_ok(new_root / "lib", spec)
    out = weights.ensure_derived(tar_path, spec, root=new_root)
    new = (f"_derived_ok on the same partial dir -> {partial_ok}, so it rebuilds under a "
           f"staging name and renames in: {len(list(out.iterdir()))} of 40 present")
    record("archive extraction (download_mols)",
           "a directory left by a killed extractall passes .exists()", old, new)


def case_extract_then_unlink(scratch: Path):
    """ensure_rfd3_weights extracted from the ckpt then unlink()ed it, gating on the
    existence of ONE of the four output files.

    extract_rfd3_weights writes token_initializer first, then the 639 MB
    diffusion_module.real_weights.pt last. A kill during that final torch.save leaves
    exactly the file the gate checks, truncated. From then on the gate short-circuits,
    the checkpoint sitting next to it is never consulted again, and the only fix is an
    rm -rf a user has no way to know about. Once a run HAS succeeded the checkpoint is
    unlinked, so any later damage to the weights directory is unrecoverable without a
    2.5 GB re-download the old gate never triggers.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    ckpt = scratch / "rfd3.ckpt"
    payload = {f"layer{i}": torch.zeros(64) for i in range(8)}
    torch.save(payload, ckpt)
    expect = ("token_initializer.real_weights.pt", "token_initializer.real_weights.meta.json",
              "diffusion_module.real_weights.pt", "diffusion_module.real_weights.meta.json")

    def killed_extraction(w: Path):
        """What is on disk after a SIGKILL during the last torch.save."""
        w.mkdir(parents=True, exist_ok=True)
        torch.save(payload, w / expect[0])
        (w / expect[1]).write_text(json.dumps({"ok": 1}))
        full = w / "tmp.pt"
        torch.save(payload, full)
        n = int(full.stat().st_size * 0.6)
        (w / expect[2]).write_bytes(full.read_bytes()[:n])   # truncated, and it exists
        full.unlink()

    old_w = scratch / "old" / "rfd3" / "weights"
    killed_extraction(old_w)
    old_ckpt = scratch / "old" / "rfd3.ckpt"
    shutil.copy2(ckpt, old_ckpt)
    old_ckpt.unlink()                       # the state after any earlier successful run
    gate = (old_w / expect[2]).exists()
    try:
        torch.load(old_w / expect[2], map_location="cpu", weights_only=True)
        loads = "torch.load unexpectedly SUCCEEDED"
    except Exception as e:
        loads = f"torch.load raises {type(e).__name__}: {str(e)[:60]}"
    old = (f"gate checks only diffusion_module.real_weights.pt: exists -> {gate}, so it "
           f"returns the directory and never rebuilds. That file is truncated to "
           f"{(old_w / expect[2]).stat().st_size} bytes and {loads}. The source ckpt is "
           f"gone (exists -> {old_ckpt.exists()}): permanent, with no path back except a "
           f"manual rm -rf.")

    spec = weights.Derived("rfd3/weights", "rfd3", expect=expect, discard_archive=True)
    new_root = scratch / "new"
    new_w = new_root / "rfd3" / "weights"
    killed_extraction(new_w)
    new_ckpt = new_root / "rfd3.ckpt"
    shutil.copy2(ckpt, new_ckpt)
    partial_ok = weights._derived_ok(new_w, spec)

    def fake_extract(archive, out):
        out.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out / expect[0])
        (out / expect[1]).write_text(json.dumps({"ok": 1}))
        torch.save(payload, out / expect[2])
        (out / expect[3]).write_text(json.dumps({"ok": 1}))

    orig = weights._produce
    weights._produce = lambda producer, archive, staging, quiet=False: fake_extract(archive, staging)
    try:
        out = weights.ensure_derived(new_ckpt, spec, root=new_root)
    finally:
        weights._produce = orig
    all_load = all(weights.artifact_intact(out / n) for n in expect)
    new = (f"_derived_ok verifies all four files, not one name: -> {partial_ok} on the same "
           f"directory. It re-extracts into staging, checks every output loads "
           f"({all_load}), renames the directory in, and only then unlinks the ckpt "
           f"(exists -> {new_ckpt.exists()}). With the ckpt already gone, fetch() "
           f"re-downloads it instead of returning a broken directory.")
    record("extract-then-discard (ensure_rfd3_weights)",
           "the one file the gate names is the last and largest one written", old, new)


def case_manual(scratch: Path):
    """OpenFold3 is never downloaded, so all we can do is verify what we are handed.
    A truncated scp used to die inside torch.load with no hint about the cause."""
    scratch.mkdir(parents=True, exist_ok=True)
    src = REAL / "boltzgen" / "boltzgen1_ifold.ckpt"
    dst = scratch / "of3-p2-155k.pt"
    truncate_copy(src, dst, 0.60)
    old = "Path(of3_ckpt).exists() -> True; the run dies later inside torch.load"
    os.environ["TT_BIO_OPENFOLD3"] = str(dst)
    try:
        weights.fetch("openfold3")
        new = "unexpectedly accepted"
    except RuntimeError as e:
        new = f"fetch() raises RuntimeError: {str(e)[:100]}"
    finally:
        del os.environ["TT_BIO_OPENFOLD3"]
    record("manual checkpoint (OpenFold3)",
           "a half-copied file is reported by torch, not by tt-bio", old, new)


def main():
    with tempfile.TemporaryDirectory(prefix="ttbio-poison-") as td:
        scratch = Path(td)
        case_flat_file(scratch / "flat")
        case_url_download(scratch / "url")
        case_tar_extract(scratch / "tar")
        case_extract_then_unlink(scratch / "unlink")
        case_manual(scratch / "manual")
    print("\n" + json.dumps(OUT, indent=2))
    Path(__file__).with_name("repro_poisoning.json").write_text(json.dumps(OUT, indent=2))


if __name__ == "__main__":
    main()

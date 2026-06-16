import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import ttnn


TTNN_ROOT = Path(ttnn.__file__).resolve().parent
SFPI_VERSION_FILE = TTNN_ROOT / "tt_metal" / "sfpi-version"
RUNTIME_SFPI = TTNN_ROOT / "runtime" / "sfpi"
SYSTEM_SFPI = Path("/opt/tenstorrent/sfpi")
KERNEL_CACHE = Path.home() / ".cache" / "tt-metal-cache"


def _sfpi_meta() -> dict:
    """Parse the ttnn wheel's ``sfpi-version`` file into a dict (version, repo,
    and the per-target package hashes)."""
    meta = {}
    for line in SFPI_VERSION_FILE.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*['\"]?([^'\"\n]*)", line)
        if m:
            meta[m.group(1)] = m.group(2)
    if "sfpi_version" not in meta:
        raise RuntimeError(f"Unable to parse sfpi_version from {SFPI_VERSION_FILE}")
    return meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_from_release(meta: dict, arch: str, distro: str, ext: str) -> None:
    """Install the EXACT pinned SFPI from the SFPI GitHub release.

    ttnn ships prebuilt kernel objects whose LTO bytecode is tied to the exact
    compiler it was built with, so a newer SFPI miscompiles them (an
    ``lto_read_decls`` internal compiler error at device bring-up). The APT repo
    prunes old point releases — their ``.deb`` 403s even while still listed — so
    when APT can't serve the pinned version we fetch that same version from the
    GitHub release, which is stable, and verify it against the wheel's hash.
    """
    version = meta["sfpi_version"]
    repo = meta.get("sfpi_repo", "https://github.com/tenstorrent/sfpi")
    asset = f"sfpi_{version}_{arch}_{distro}.{ext}"
    url = f"{repo}/releases/download/{version}/{asset}"
    expected = meta.get(f"sfpi_{arch}_{distro}_{ext}_hash")
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / asset
        print(f"Fetching SFPI {version} from {url}")
        urllib.request.urlretrieve(url, pkg)
        if expected and _sha256(pkg) != expected:
            raise RuntimeError(f"SFPI {asset} hash mismatch — refusing to install.")
        if ext == "deb":
            subprocess.check_call(["sudo", "dpkg", "-i", str(pkg)])
        else:
            subprocess.check_call(["sudo", "rpm", "-i", "--force", str(pkg)])


def _install_sfpi(meta: dict) -> None:
    version = meta["sfpi_version"]
    arch = os.uname().machine
    if shutil.which("apt-get"):
        distro, ext = "debian", "deb"
        # Fast path: APT may still serve this exact version (handles deps too).
        try:
            subprocess.check_call(
                ["sudo", "apt-get", "install", "-y", "--allow-downgrades", f"sfpi={version}"])
            return
        except subprocess.CalledProcessError:
            pass
        _install_from_release(meta, arch, distro, ext)
        return

    if shutil.which("dnf"):
        distro, ext = "fedora", "rpm"
        try:
            subprocess.check_call(["sudo", "dnf", "install", "-y", f"sfpi-{version}"])
            return
        except subprocess.CalledProcessError:
            pass
        _install_from_release(meta, arch, distro, ext)
        return

    raise RuntimeError(f"Unsupported package manager. Install sfpi {version} manually.")


def _use_system_sfpi() -> None:
    if not RUNTIME_SFPI.exists() and not RUNTIME_SFPI.is_symlink():
        return

    if not RUNTIME_SFPI.is_symlink():
        print(f"Keeping bundled SFPI at {RUNTIME_SFPI}")
        return

    if RUNTIME_SFPI.resolve(strict=False) == SYSTEM_SFPI:
        return

    RUNTIME_SFPI.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_SFPI.unlink(missing_ok=True)
    RUNTIME_SFPI.symlink_to(SYSTEM_SFPI, target_is_directory=True)
    print(f"Linked {RUNTIME_SFPI} -> {SYSTEM_SFPI}")


def main() -> None:
    meta = _sfpi_meta()
    print(f"Installing SFPI {meta['sfpi_version']} for ttnn at {TTNN_ROOT}")
    _install_sfpi(meta)
    _use_system_sfpi()

    shutil.rmtree(KERNEL_CACHE, ignore_errors=True)
    print(f"Cleared {KERNEL_CACHE}")


if __name__ == "__main__":
    main()

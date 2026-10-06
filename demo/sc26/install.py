#!/usr/bin/env python3
"""Set up a QuietBox 2 so it runs the SC26 booth demo exactly as the QuietBox it was built on.

    ~/sc26/demo/sc26/install.py           check, fix what is safe, install the demo
    ~/sc26/demo/sc26/install.py --check   check only, change nothing

What "exactly" means is install/system.json: kernel, driver, firmware, the Tenstorrent packages,
the Python packages (install/*.lock), the model weights and the booth session. The script checks
everything first and prints what it found. Then it fixes what it can: the firmware it never
touches, it only tells you the command. Running it again is safe; a box that already matches is
left alone and the script says so.

Options:
    --chips 0,1,2,3       chips the demo folds on (default: all four)
    --accept-firmware     carry on although the board firmware differs (untested at the booth)
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parents[1]
OPS = DEMO / "ops"
SPEC = json.loads((DEMO / "install" / "system.json").read_text())
HOME = Path.home()
ENVS = HOME / "sc26-env"
ENGINE_PY = ENVS / "engine" / "bin" / "python3"
TOOLS = ENVS / "tools" / "bin"
CFG = HOME / ".config" / "sc26" / "env"
DOWNLOADS = HOME / "sc26-downloads"


def run(cmd, sudo=False, check=True, quiet=False, **kw):
    """Run a command, echoing it first so the person at the keyboard sees every change."""
    cmd = (["sudo"] if sudo else []) + [str(c) for c in cmd]
    if not quiet:
        print("    $ " + shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=check, **kw)


def out(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def read(p) -> str:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return ""


def download(url, sha256) -> Path:
    """Fetch a release file once into ~/sc26-downloads and refuse it unless its hash matches."""
    DOWNLOADS.mkdir(exist_ok=True)
    dest = DOWNLOADS / url.rsplit("/", 1)[1]
    if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
        print(f"    downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dest)
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != sha256:
            dest.unlink()
            sys.exit(f"{dest.name}: sha256 {got}, expected {sha256}. Not using it.")
    return dest


def dpkg_version(pkg) -> str:
    v = out(["dpkg-query", "-W", "-f=${db:Status-Abbrev} ${Version}", pkg]).split()
    return v[1] if len(v) == 2 and v[0][1:2] == "i" else ""   # ii, or hi when held


def apt_install(*pkgs):
    run(["apt-get", "update", "-q"], sudo=True)
    run(["apt-get", "install", "-y", "-q", "--allow-downgrades", "--allow-change-held-packages", *pkgs], sudo=True)


def chips() -> list[Path]:
    return sorted(Path("/sys/class/tenstorrent").glob("*"))


@dataclass
class Item:
    name: str
    want: str
    have: str
    fix: Callable | None = None   # None: this script cannot fix it
    reboot: bool = False          # the fix takes effect at the next boot

    @property
    def ok(self):
        return self.have == self.want


# ---- the system ---------------------------------------------------------------------------

def os_release():
    r = dict(l.split("=", 1) for l in read("/etc/os-release").splitlines() if "=" in l)
    have = f"{r.get('ID', '?')} {r.get('VERSION_ID', '?').strip(chr(34))}"
    return Item("Operating system", f"{SPEC['os']['id']} {SPEC['os']['version']}", have)


def packages():
    want = SPEC["apt"]
    missing = [p for p in want if not dpkg_version(p)]
    return Item("Desktop packages", "installed", "missing: " + " ".join(missing) if missing else "installed",
                lambda: apt_install(*missing))


def ppa():
    want = SPEC["ppa"]["list"]
    have = read("/etc/apt/sources.list.d/tenstorrent.list")

    def fix():
        run(["install", "-d", "-m", "0755", "/etc/apt/keyrings"], sudo=True)
        run(["curl", "-fsSL", "-o", "/etc/apt/keyrings/tt-pkg-key.asc", SPEC["ppa"]["key"]], sudo=True)
        run(["sh", "-c", f"echo '{want}' > /etc/apt/sources.list.d/tenstorrent.list"], sudo=True)
    return Item("Tenstorrent package source", "ppa.tenstorrent.com",
                "ppa.tenstorrent.com" if "ppa.tenstorrent.com" in have else "not set up", fix)


def debs():
    want = SPEC["debs"]
    have = {p: dpkg_version(p) for p in want}
    fmt = lambda d: ", ".join(f"{p} {v or 'missing'}" for p, v in d.items())

    def fix():
        apt_install(*[f"{p}={v}" for p, v in want.items() if have[p] != v])
        # The compiler the chips' kernels are built with must match the ttnn wheel, so hold it.
        run(["apt-mark", "hold", "sfpi"], sudo=True)
        run(["systemctl", "enable", "tenstorrent-hugepages.service"], sudo=True)
    # tenstorrent-tools sets up the hugepages the chips need; it takes effect at boot.
    return Item("Tenstorrent packages", fmt(want), fmt(have), fix, reboot=True)


def kernel():
    want = SPEC["kernel"]["release"]
    have = os.uname().release

    def fix():
        apt_install(*SPEC["kernel"]["packages"])
        # Boot this kernel by default even when a newer one is installed. GRUB's own ids for the
        # entry carry the root file system's UUID.
        uuid = out(["findmnt", "-no", "UUID", "/"])
        entry = f"gnulinux-advanced-{uuid}>gnulinux-{want}-advanced-{uuid}"
        run(["sh", "-c", f"echo 'GRUB_DEFAULT=\"{entry}\"' > /etc/default/grub.d/99-sc26-kernel.cfg"], sudo=True)
        run(["update-grub"], sudo=True)
    return Item("Linux kernel", want, have, fix, reboot=True)


def kmd():
    want = SPEC["kmd"]["version"]
    loaded = read("/sys/module/tenstorrent/version")
    # The module the booth kernel will load, which is the one that matters after a reboot.
    boot = out(["modinfo", "-k", SPEC["kernel"]["release"], "-F", "version", "tenstorrent"])
    have = loaded if loaded == boot else f"{loaded or 'not loaded'} (next boot: {boot or 'none'})"

    def fix():
        deb = download(SPEC["kmd"]["deb"], SPEC["kmd"]["sha256"])
        run(["apt-get", "install", "-y", "-q", "--allow-downgrades", deb], sudo=True)
    return Item("Tenstorrent driver (tt-kmd)", want, have, fix, reboot=True)


def kmd_options():
    want = " ".join(f"{k}={v}" for k, v in SPEC["kmd"]["options"].items())
    conf = " ".join(l for l in out(["modprobe", "-c"]).splitlines() if l.startswith("options tenstorrent "))
    live = {k: read(f"/sys/module/tenstorrent/parameters/{k}") for k in SPEC["kmd"]["options"]}
    configured = all(f"{k}={v}" in conf for k, v in SPEC["kmd"]["options"].items())
    # Booleans read back as Y/N; power_policy=0 reads N.
    active = all(live[k] in (v, {"0": "N", "1": "Y"}.get(v)) for k, v in SPEC["kmd"]["options"].items())
    have = want if configured and active else (f"{want} after reboot" if configured else conf or "default")

    def fix():
        run(["sh", "-c", f"echo 'options tenstorrent {want}' > /etc/modprobe.d/sc26-tenstorrent.conf"], sudo=True)
    return Item("Driver options", want, have, None if configured else fix, reboot=True)


def chip_count():
    n = len(out(["lspci", "-d", "1e52:"]).splitlines())
    return Item("Blackhole chips on PCIe", str(SPEC["chips"]["count"]), str(n))


def hugepages():
    want = SPEC["chips"]["count"] * SPEC["chips"]["hugepages_1g_per_chip"]
    have = int(read("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages") or 0)
    # tenstorrent-tools reserves them at boot (tenstorrent-hugepages.service).
    return Item("1 GB hugepages", f"{want} or more", f"{want} or more" if have >= want else str(have),
                None, reboot=True)


def firmware():
    want = SPEC["firmware"]["bundle_version"]
    vers = {c.name.split("!")[-1]: read(c / "tt_fw_bundle_ver") for c in chips()}
    if not vers:
        have = "unknown until the driver is loaded"
    elif set(vers.values()) == {want}:
        have = want
    else:
        have = ", ".join(f"chip {k}: {v or '?'}" for k, v in vers.items())
    types = {read(c / "tt_card_type") for c in chips()}
    if types and types != {SPEC["chips"]["card_type"]}:
        have += f" (cards: {', '.join(sorted(types))})"
    return Item("Board firmware", want, have)


def watchdog():
    want = SPEC["watchdog"]["runtime_usec"]
    show = dict(l.split("=", 1) for l in out(["systemctl", "show", "-p", "RuntimeWatchdogUSec",
                                              "-p", "WatchdogDevice"]).splitlines() if "=" in l)
    have = show.get("RuntimeWatchdogUSec", "") if show.get("WatchdogDevice") else "off"
    installed = Path("/etc/systemd/system.conf.d/99-sc26-watchdog.conf").exists() or have == want

    def fix():
        sysdir = DEMO / "install" / "system"
        for src, dst in (("sp5100-tco.conf", "/etc/modprobe.d/sc26-sp5100-tco.conf"),
                         ("sc26-watchdog-modules.conf", "/etc/modules-load.d/sc26-watchdog.conf"),
                         ("99-sc26-watchdog.conf", "/etc/systemd/system.conf.d/99-sc26-watchdog.conf"),
                         ("sc26-hw-watchdog.service", "/etc/systemd/system/sc26-hw-watchdog.service")):
            run(["install", "-D", "-m", "0644", sysdir / src, dst], sudo=True)
        run(["install", "-D", "-m", "0755", sysdir / "sc26-arm-hw-watchdog.sh",
             "/usr/local/sbin/sc26-arm-hw-watchdog.sh"], sudo=True)
        run(["systemctl", "daemon-reload"], sudo=True)
        run(["systemctl", "enable", "sc26-hw-watchdog.service"], sudo=True)
    if installed and have != want:
        have = f"{have}, on after reboot"
    return Item("Hardware watchdog", want, have, None if installed else fix, reboot=True)


def browser():
    have = "installed" if out(["snap", "list", "firefox"]) else "missing"
    return Item("Firefox", "installed", have, lambda: run(["snap", "install", "firefox"], sudo=True))


# ---- the demo -----------------------------------------------------------------------------

def lock_env(name, lock, extra=()):
    py = ENVS / name / "bin" / "python3"
    want = sorted(l for l in read(DEMO / "install" / lock).splitlines() if l and not l.startswith("#"))
    have = sorted(l for l in out([py, "-m", "pip", "freeze", "--exclude-editable"]).splitlines()
                  if "==" in l) if py.exists() else []
    differ = len(set(want) ^ set(have))

    def fix():
        run([f"python{SPEC['python']}", "-m", "venv", "--clear", ENVS / name])
        # --no-deps: the lock is the whole environment, resolved once on the booth QuietBox.
        run([py, "-m", "pip", "install", "-q", "--no-deps", *extra, "-r", DEMO / "install" / lock])
    label = f"{len(want)} packages as locked"
    have_s = label if not differ else ("missing" if not have else f"{differ} packages differ")
    return Item(f"Python {name} environment", label, have_s, fix)


def weights():
    want = SPEC["weights"]
    probe = ("import sys, json; sys.path.insert(0, sys.argv[1]); from tt_bio import weights as w; "
             "print(json.dumps({k: w.status(k).state for k in sys.argv[2:]}))")
    try:
        st = json.loads(out([ENGINE_PY, "-c", probe, REPO, *want]).splitlines()[-1])
    except (IndexError, ValueError):
        st = {}
    missing = [k for k in want if st.get(k) != "present"]

    def fix():
        # tt-bio's own fetch, verified. The booth folds with OpenFold3 only: 2.3 GB.
        run([ENGINE_PY, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from tt_bio import weights as w; "
             "[w.fetch(k) for k in sys.argv[2:]]", REPO, *missing])
    have = "all present" if not missing else "missing: " + " ".join(missing)
    return Item("Model weights", "all present", have, fix)


def gallery():
    n = len(json.loads(read(DEMO / "gallery" / "manifest.json"))["entries"])
    have_n = len(list((DEMO / "gallery" / "trajectories").glob("*.jsonl")))
    return Item("Recorded folds", f"{n} recordings", f"{have_n} recordings",
                lambda: run([ENGINE_PY, DEMO / "gallery" / "build.py", "--check"]))


def config(chip_list):
    """~/.config/sc26/env: which Python and tt-smi the demo uses, and which chips it folds on.
    Other lines (an out-of-service chip, say) are kept."""
    want = {"SC26_PYTHON": str(ENGINE_PY), "TT_SMI": str(TOOLS / "tt-smi")}
    if chip_list or not CFG.exists():
        want["SC26_CHIPS"] = chip_list or ",".join(map(str, range(SPEC["chips"]["count"])))
    lines = read(CFG).splitlines()
    have = dict(l.split("=", 1) for l in lines if "=" in l and not l.startswith("#"))
    bad = [k for k, v in want.items() if have.get(k) != v]

    def fix():
        keep = [l for l in lines if l.split("=", 1)[0] not in want]
        CFG.parent.mkdir(parents=True, exist_ok=True)
        CFG.write_text("\n".join(keep + [f"{k}={v}" for k, v in want.items()]) + "\n")
        print(f"    wrote {CFG}")
    return Item("Demo settings", "set", "to write: " + " ".join(bad) if bad else "set", fix)


def session():
    unit = read(HOME / ".config/systemd/user/sc26-engine.service")
    have = "installed" if str(OPS) in unit and Path("/usr/share/wayland-sessions/sc26.desktop").exists() \
        and f"AutomaticLogin={os.environ.get('USER', '')}" in read("/etc/gdm3/custom.conf").replace(" ", "") \
        else "not installed"
    # sc26ctl install: user units, autologin into the bare sway session, kiosk policies, the
    # power button reboots, no automatic upgrades during the show.
    return Item("Booth session (boots into the demo)", "installed", have,
                lambda: run([OPS / "sc26ctl", "install"], env=dict(os.environ, SC26_PYTHON=str(ENGINE_PY))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="check only, change nothing")
    ap.add_argument("--chips", default="", help="chips the demo folds on, e.g. 0,1,3 (default: all four)")
    ap.add_argument("--accept-firmware", action="store_true",
                    help="carry on although the board firmware differs from the booth's")
    a = ap.parse_args()

    system = [os_release, packages, ppa, debs, kernel, kmd, kmd_options, hugepages, watchdog, browser,
              chip_count, firmware]
    demo = [lambda: lock_env("tools", "tools.lock"),
            lambda: lock_env("engine", "engine.lock", ("--extra-index-url", SPEC["torch_index"])),
            weights, gallery, lambda: config(a.chips), session]

    print(f"SC26 booth demo install, from {REPO} ({out(['git', '-C', REPO, 'describe', '--tags', '--always'])})\n")
    items = [f() for f in system]
    report(items)
    fw = next(i for i in items if i.name == "Board firmware")
    if not fw.ok and fw.have.startswith("chip") and not a.accept_firmware:
        firmware_help(fw)
        return 2
    stuck = [i for i in items if not i.ok and i.fix is None and not i.reboot and i is not fw]
    if a.check:
        report([f() for f in demo], header=False)
        return 0 if all(i.ok for i in items) else 1
    if stuck:
        print("\nThis box differs from the booth QuietBox in a way this script cannot fix:")
        for i in stuck:
            print(f"  {i.name}: has {i.have}, needs {i.want}")
        print("Call the contact in INSTALL.md before going on.")
        return 1

    if any(not i.ok and i.fix for i in items):
        print("\nFixing the system (sudo may ask for your password once):")
        run(["sudo", "-v"], quiet=True)
    changed, reboot = False, False
    for i in items:
        if not i.ok and i.fix:
            if not apply(i):
                return 1
            changed, reboot = True, reboot or i.reboot
    print()
    for f in demo:
        i = f()
        if i.ok:
            print(f"  ok      {i.name}: {i.have}")
            continue
        if not apply(i):
            return 1
        changed = True
        i = f()
        if not i.ok:
            print(f"\n{i.name} still reads {i.have} after the fix. Stopping here; see INSTALL.md, 'If something goes wrong'.")
            return 1

    after = [f() for f in system]
    print()
    if not changed:
        print("Nothing to change: this box matches the booth QuietBox.")
    if reboot or any(not i.ok for i in after):
        print("Reboot now to finish:  sudo reboot\n"
              "The box comes back straight into the demo. Run this script once more after the reboot;\n"
              "it should end with 'Nothing to change'.")
    elif changed:
        print("Installed. Start the demo now with  ~/sc26/demo/sc26/ops/sc26ctl start  or reboot into it.")
    return 0


def apply(i) -> bool:
    """Make one change, timed. A failed command ends the run with a plain message, no traceback."""
    print(f"  change  {i.name}: {i.have} -> {i.want}", flush=True)
    t = time.monotonic()
    try:
        i.fix()
    except subprocess.CalledProcessError as e:
        print(f"\n{i.name}: `{shlex.join(map(str, e.cmd))}` failed (exit {e.returncode}).\n"
              "Nothing after it was changed. Run this script again; if it stops at the same place, "
              "see INSTALL.md, 'If something goes wrong'.")
        return False
    print(f"          done in {time.monotonic() - t:.0f} s", flush=True)
    return True


def report(items, header=True):
    if header:
        print(f"  {'':8}{'what':38}{'this box':44}booth QuietBox")
    for i in items:
        mark = "ok" if i.ok else ("fix" if i.fix else ("reboot" if i.reboot else "DIFFERS"))
        print(f"  {mark:8}{i.name:38}{i.have[:43]:44}{i.want}")


def firmware_help(fw):
    f = SPEC["firmware"]
    print(f"""
STOPPED: the board firmware differs from the booth QuietBox, and this script never flashes it.
  this box: {fw.have}
  booth:    {f['bundle_version']} on every chip

Nothing has been changed. Flashing writes each board's boot flash. If the power drops or the
command is interrupted half way, the board can be left unable to start, so do it only when
nothing else is running on the box and leave it alone until it says it is done.

  1. Download and check the firmware bundle (Tenstorrent's public release):
       mkdir -p ~/sc26-downloads && cd ~/sc26-downloads
       curl -fLO {f['url']}
       echo "{f['sha256']}  {f['url'].rsplit('/', 1)[1]}" | sha256sum -c
  2. Flash it (a few minutes for both boards; do not interrupt it):
       ~/sc26-env/tools/bin/tt-flash flash ~/sc26-downloads/{f['url'].rsplit('/', 1)[1]}
     If ~/sc26-env/tools does not exist yet, use the tt-flash that came with the box.
     If tt-flash refuses because the board has NEWER firmware, do not force it. Either ask the
     contact in INSTALL.md, or run this script with --accept-firmware to keep the newer firmware
     (the demo has not been tested on it).
  3. Reboot (sudo reboot), then run this script again.
""")


if __name__ == "__main__":
    sys.exit(main())

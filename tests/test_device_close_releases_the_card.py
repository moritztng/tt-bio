"""A closed card must be a free card: no driver fds left, and the next process can open it.

`ttnn.close_device()` alone does not do this. tt-metal's own `CloseDevices` docstring says
so: "After this call, this process still controls all devices. Call ReleaseOwnership() to
fully release ownership." Without that second call the MetalContext singleton survives, and
with it the UMD cluster, its two /dev/tenstorrent/<N> fds and the CHIP_IN_USE robust mutex,
until the process exits -- so `cleanup()` advertised the card as free while still holding
the chip, and the next process to open it blocked in futex_wait inside
`LocalChip::start_device` and never returned.

That is a whole-suite hazard and not one test's problem: any collection order that puts an
in-process device test before a device-spawning one used to stall the run (measured: this
file's two-command ancestor ran 400 s to a timeout kill, and passes in 6 s now). ttnn does
not bind ReleaseOwnership, so `tenstorrent._close_device_locked` calls the exported symbol
directly -- which is exactly the kind of dependence on a wheel internal that has to be
pinned by a test rather than trusted.
"""
import glob
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

pytestmark = pytest.mark.device


def _driver_fds():
    """Open fds of this process pointing at a TT card."""
    out = []
    for p in glob.glob("/proc/self/fd/*"):
        try:
            target = os.readlink(p)
        except OSError:      # the fd went away while we were listing
            continue
        if target.startswith("/dev/tenstorrent/"):
            out.append(target)
    return sorted(out)


_CHILD = r"""
import os, sys
sys.path.insert(0, os.environ["REPO"])
import torch, ttnn
from tt_bio.tenstorrent import get_device
dev = get_device()
x = ttnn.from_torch(torch.ones((32, 32), dtype=torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=dev)
print("CHILD_OK " + str(float(ttnn.to_torch(ttnn.add(x, x))[0, 0])))
"""


def test_cleanup_drops_the_driver_fds():
    """The mechanism, read straight off /proc: no fd on the card survives cleanup()."""
    from tt_bio import tenstorrent as tt

    assert _driver_fds() == [], f"a card was already open before this test: {_driver_fds()}"
    tt.get_device()
    assert _driver_fds(), "get_device() opened no fd on a card, so this proves nothing"
    tt.cleanup()
    assert _driver_fds() == [], (
        f"driver fds survived cleanup(): {_driver_fds()}. close_device() does not release "
        "ownership of the chip; the next process to open this card will hang in futex_wait "
        "inside UMD start_device. Check that _close_device_locked still resolves "
        "tt::tt_metal::detail::ReleaseOwnership in this ttnn wheel.")


def test_a_child_can_open_the_card_this_process_just_closed():
    """The consequence, which is what the suite actually needs: open, close, hand over.

    Bounded on purpose. A regression here is a hang, not a wrong answer, so it has to
    fail on the clock rather than take the run down with it.
    """
    from tt_bio import tenstorrent as tt

    tt.get_device()
    tt.cleanup()
    try:
        r = subprocess.run([sys.executable, "-c", _CHILD], env=dict(os.environ, REPO=REPO),
                           cwd=REPO, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail("a child process could not open the card this process closed: it hung. "
                    "cleanup() is releasing the card lease while still holding the chip.")
    assert r.returncode == 0, f"child failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    assert "CHILD_OK 2.0" in r.stdout, f"child opened the card but miscomputed:\n{r.stdout[-2000:]}"


def test_reopening_after_cleanup_still_works_in_this_process():
    """Releasing ownership must not be one-way: MetalContext is recreated on next access.

    Every model funnels through cleanup(), and the platform's worker pool closes and
    reopens within one process, so a teardown that could only be done once would break it.
    """
    import torch
    import ttnn
    from tt_bio import tenstorrent as tt

    tt.get_device()
    tt.cleanup()
    dev = tt.get_device()
    x = ttnn.from_torch(torch.ones((32, 32), dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    assert float(ttnn.to_torch(ttnn.add(x, x))[0, 0]) == 2.0
    tt.cleanup()

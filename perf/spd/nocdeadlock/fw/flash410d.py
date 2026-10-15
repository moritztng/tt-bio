"""flash410d.py [--probe] <tt-flash args...>: tt-flash restricted to PCI interfaces 0 and 1 (qb2 board 410D).
tt-flash 4.0 flashes and resets every detected chip and ignores TT_VISIBLE_DEVICES; this filters both detection
paths (pre-flash and post-reset) so board 4103 (chips 2/3) is never touched."""
import sys
import tt_flash.chip as c, tt_flash.flash as f, tt_flash.main as m
KEEP = {0, 1}
_dl, _dc = c.detect_local_chips, c.detect_chips
m.detect_local_chips = lambda *a, **k: [d for d in _dl(*a, **k) if d.interface_id in KEEP]
f.detect_chips = lambda *a, **k: [d for d in _dc(*a, **k) if d.interface_id in KEEP]
if sys.argv[1:2] == ["--probe"]:
    for d in m.detect_local_chips(ignore_ethernet=True):
        print(d.interface_id, type(d).__name__, c.resolve_board_type(d) if hasattr(c, "resolve_board_type") else "", d.get_telemetry_unchanged().board_id if hasattr(d, "get_telemetry_unchanged") else "")
    sys.exit(0)
sys.argv = ["tt-flash"] + sys.argv[1:]
sys.exit(m.main())

"""What the lever census can SEE, tested statically without a device.

The census answers one question for the whole fleet: does a shipped lever actually fire, or is it
merely present and switched on? It can only answer that for a lever it has a row for, and a lever
with no row is not reported as uncovered -- it is simply absent from the table, which reads like a
lever that does not exist. The three eltwise fusions shipped that way and stayed invisible until
2026-09-20, `scale_add` among them, which declines on its operands' dtype: a property of the call
site that no amount of reading the default can tell you.

Two things are pinned here.

`test_every_row_resolves` -- a row naming a flag or counter the tree no longer defines would report
that lever as dark forever, and a rename is the ordinary way that happens.

`test_counter_frontier` -- the guard against the next blind spot. 14 two-slot counters have no
census row today. Some are not levers at all, so asserting full coverage would be asserting
something untrue; instead the frontier is frozen by name. Adding a counter without a row fails this
test, and the author either gives it a row or adds it here on purpose. Growth stays deliberate.
"""
import ast
import pathlib
import re

WT = pathlib.Path(__file__).resolve().parent.parent

# Counters with no census row as of 2026-09-20. Not a wishlist: several are fallbacks or
# diagnostics rather than levers. To add to this list, say in the commit why the counter is not
# a lever the census should report.
UNCENSUSED = {
    "tt_bio.esmc.WINDOW_FALLBACK_STATS",
    "tt_bio.protenix.RELP_STATS",
    "tt_bio.rfd3_bias.DSTATS",
    "tt_bio.tenstorrent.ATOM_SHIFT_GATHER_STATS",
    "tt_bio.tenstorrent.OPM_SMALL_DEPTH_STATS",
    "tt_bio.tenstorrent.PWA_BATCH_HEAD_STATS",
    "tt_bio.tenstorrent.RESIDUAL_L1_STATS",
    "tt_bio.tenstorrent.SDPA_RAGGED_PAD_STATS",
    "tt_bio.tenstorrent.TRIMUL_GOUT_STATS",
    "tt_bio.tenstorrent.TRIMUL_MASK_L1_STATS",
    "tt_bio.triatt_qkv.QKVGB_STATS",
    "tt_bio.triatt_qkv.QKVG_STATS",
    "tt_bio.triatt_sdpa.GATE_STATS",
    "tt_bio.trimul_tail.OUT_L1_STATS",
}


def _rows():
    src = (WT / "scripts" / "lever_census.py").read_text()
    body = re.search(r"^LEVERS = \[(.*?)^\]", src, re.M | re.S).group(1)
    return ast.literal_eval("[" + body + "]")


def _assigns(dotted):
    """True if `dotted` (tt_bio.mod.NAME) is assigned at module scope in the tree."""
    mod, _, name = dotted.rpartition(".")
    f = WT / (mod.replace(".", "/") + ".py")
    return f.is_file() and re.search(rf"^{re.escape(name)}\b", f.read_text(), re.M) is not None


def test_every_row_resolves():
    bad = []
    for name, mod, flag, stats, _kind in _rows():
        if not _assigns(f"{mod}.{flag}"):
            bad.append(f"{name}: flag {mod}.{flag} is not assigned in the tree")
        if stats and not _assigns(stats.split(":")[0]):
            bad.append(f"{name}: counter {stats} is not assigned in the tree")
    assert not bad, "census rows pointing at symbols the tree no longer has:\n" + "\n".join(bad)


def test_lever_names_are_unique():
    names = [r[0] for r in _rows()]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate lever names collide in the report: {sorted(dupes)}"


def test_counter_frontier():
    """Every two-slot [served, declined] counter is either censused or listed above."""
    censused = {r[3].split(":")[0] for r in _rows() if r[3]}
    found = set()
    for f in sorted((WT / "tt_bio").glob("*.py")):
        for m in re.finditer(r"^([A-Z][A-Z0-9_]*STATS[A-Z0-9_]*)\s*=\s*\[\s*0\s*,\s*0\s*\]",
                             f.read_text(), re.M):
            found.add(f"tt_bio.{f.stem}.{m.group(1)}")
    unseen = found - censused - UNCENSUSED
    assert not unseen, (
        "counters the census cannot see and that are not on the known frontier:\n  "
        + "\n  ".join(sorted(unseen))
        + "\nGive each a row in scripts/lever_census.py LEVERS, or add it to UNCENSUSED with the "
          "reason it is not a lever.")
    stale = UNCENSUSED - found
    assert not stale, (
        "UNCENSUSED names counters that no longer exist -- drop them so the list keeps meaning "
        f"something: {sorted(stale)}")


def test_the_three_eltwise_fusions_are_counted():
    """The blind spot that motivated this file: a shipped lever with no counter at all."""
    src = (WT / "tt_bio" / "eltwise_fusion.py").read_text()
    censused = {r[3].split(":")[0] for r in _rows() if r[3]}
    for flag, counter in (("FUSE_SCALE_ADD", "SCALE_ADD_STATS"),
                          ("FUSE_MASK_ADD", "MASK_ADD_STATS"),
                          ("FUSE_NORM_RESIDUAL", "NORM_RESIDUAL_STATS")):
        assert re.search(rf"^{flag}\s*=\s*env_flag", src, re.M), f"{flag} is not a flag any more"
        assert re.search(rf"^{counter}\s*=\s*\[0, 0\]", src, re.M), f"{flag} lost its counter"
        assert src.count(f"{counter}[0] += 1") == 1, f"{counter} does not count the served path"
        assert src.count(f"{counter}[1] += 1") == 1, f"{counter} does not count the declined path"
        assert f"tt_bio.eltwise_fusion.{counter}" in censused, f"{flag} has no census row"

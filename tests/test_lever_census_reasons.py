"""The census reject-reason emit, tested without a device.

A lever that reads served=0 is either dark or correctly declining, and only the reason tells
them apart. These are the four cells the 2026-08-19 size recheck turned on: the tri-bias
projection (8,1) that no _MM_BLOCK entry covers, openfold3 at c_z=128 that the kt==8 scope
refuses, an in-scope protenix pair projection, and the cross-process reason merge.

Everything happens inside run_checks(). The stand-in ttnn and the scripts/ path entry are
process-global, and pytest imports this module during collection, so installing either at
module scope hands every later test in the session a five-attribute ttnn.
"""
import json, sys, types, tempfile, pathlib

WT = str(pathlib.Path(__file__).resolve().parent.parent)
SCRIPTS = WT + "/scripts"


def _stub_ttnn():
    tt = types.ModuleType("ttnn")
    tt.bfloat16 = "bf16"; tt.float32 = "fp32"
    tt.TILE_LAYOUT = "TILE"; tt.ROW_MAJOR_LAYOUT = "RM"
    class BT: DRAM = "DRAM"; L1 = "L1"
    tt.BufferType = BT
    return tt


class W:
    def __init__(self, k, n, dtype="bf16"): self.shape = (k, n); self.dtype = dtype
class X:
    def __init__(self, *dims, dtype="bf16"): self.shape = dims; self.dtype = dtype

class T:                                     # stand-in tt_bio.tenstorrent
    _PAIR_PROJ_MM = True
    _MM_CFG = True
    _MM_DEFAULT = (8, 8, 8, 2, 2)
    _MM_BLOCK = {(8, 24): (4, 8, 1, 4, 1), (8, 8): (4, 8, 1, 4, 1),
                 (4, 12): (4, 4, 1, 4, 1), (4, 4): (4, 4, 1, 4, 1)}
    @staticmethod
    def _mm_block_for(w):
        return T._MM_BLOCK.get(((int(w.shape[-2]) + 31) // 32, (int(w.shape[-1]) + 31) // 32))


def run_checks() -> bool:
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(("PASS " if good else "FAIL ") + label + ": " + repr(got) +
              (" != " + repr(want) if not good else ""))

    # lever_census does `import ttnn` inside the function under test, so the stub has to stay
    # registered for the whole check block, then come back out.
    had = "ttnn" in sys.modules
    prev = sys.modules.get("ttnn")
    sys.modules["ttnn"] = _stub_ttnn()
    sys.path.insert(0, SCRIPTS)
    try:
        import lever_census as LC

        # the triangle-bias projection: [N,N,256] x [256,8] -> kt=8, nt=1, no _MM_BLOCK entry.
        # This is the claim that PAIR_PROJ_MINIMAL_MATMUL's 1208/440 declines are structural.
        check("tri-bias proj (8,1)", LC._pp_reason(T, "PAIR_PROJ_MINIMAL_MATMUL",
              X(512, 512, 256), W(256, 8)), "no_mm_block:(8,1)")
        # openfold3 at c_z=128: kt=4, so the kt==8 scope refuses before the table is consulted
        check("of3 c_z=128 (4,4)", LC._pp_reason(T, "PAIR_PROJ_MINIMAL_MATMUL",
              X(512, 512, 128), W(128, 128)), "k_tiles=4:(4,4)")
        # a real protenix pair projection that IS in scope reaches the op
        check("protenix pair (8,8)", LC._pp_reason(T, "PAIR_PROJ_MINIMAL_MATMUL",
              X(512, 512, 256), W(256, 256)), "op_threw:(8,8)")
        check("flag off", LC._pp_reason(type("t", (T,), {"_PAIR_PROJ_MM": False}),
              "PAIR_PROJ_MINIMAL_MATMUL", X(512, 512, 256), W(256, 8)), "flag_off")
        check("dtype", LC._pp_reason(T, "PAIR_PROJ_MINIMAL_MATMUL",
              X(512, 512, 256, dtype="fp32"), W(256, 8)), "dtype:(8,1)")

        # collect() must sum reason dicts across processes and survive tuple keys / None
        d = pathlib.Path(tempfile.mkdtemp())
        for i, rej in enumerate(({"l1_dest_is_faster:512x512x256": 600},
                                 {"l1_dest_is_faster:512x512x256": 608, "layout:1x1": 2})):
            (d / f"pid{i}.json").write_text(json.dumps({"pid": i, "argv": [], "rows": {
                "PAIR_TRANSPOSE_VIA_ROW_MAJOR": {"resolved": "True", "served": 0,
                                                 "declined": 600 + i * 10, "rejects": rej}}}))
        out = LC.collect(d, "t", [], 0)
        row = [r for r in out["rows"] if r["flag"] == "PAIR_TRANSPOSE_VIA_ROW_MAJOR"][0]
        check("collect sums declines", row["declined"], 1210)
        check("collect sums reasons", row["rejects"],
              {"l1_dest_is_faster:512x512x256": 1208, "layout:1x1": 2})
        # a lever with no reasons must still emit a well-formed row
        other = [r for r in out["rows"] if r["flag"] == "TRIMUL_IN_PROJ_DUAL_NOC"][0]
        check("absent lever row", (other["served"], other["rejects"]), (None, None))
    finally:
        sys.modules.pop("lever_census", None)
        if SCRIPTS in sys.path:
            sys.path.remove(SCRIPTS)
        if had:
            sys.modules["ttnn"] = prev
        else:
            sys.modules.pop("ttnn", None)

    print("\nALL OK" if ok else "\nFAILURES")
    return ok


def test_lever_census_reasons():
    assert run_checks(), "a check above printed FAIL"


def test_ttnn_is_not_left_stubbed():
    """The stub must not outlive run_checks(): a real ttnn is a large module, the stub has five
    attributes, and every later test in the session shares this sys.modules."""
    run_checks()
    tt = sys.modules.get("ttnn")
    assert tt is None or hasattr(tt, "get_max_worker_l1_unreserved_size"), \
        "run_checks() left its stand-in ttnn registered in sys.modules"


def _lever_census():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lever_census_under_test", WT + "/scripts/lever_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_grid_stamp_is_none_until_a_device_was_opened(monkeypatch):
    """COMPUTE_GRID_MAIN is a valid grid from import, 11x10, not a sentinel, so reading it
    straight stamped 11x10 on every process in the fold including the ones that only imported
    the module. A fold's grids are unioned across its processes, so a fold measured wholly on a
    13x10 card came out "11x10/13x10", and the size-ladder check refuses a cross-grid
    comparison outright: two p150a cells carry that string and are as unmeasured at today's
    grid as the ones that say 13x10.
    """
    lc = _lever_census()
    stub = types.ModuleType("tt_bio.tenstorrent")
    stub.COMPUTE_GRID_MAIN = (11, 10)
    monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", stub)

    stub.COMPUTE_GRID_MEASURED = False
    assert lc._compute_grid() is None
    stub.COMPUTE_GRID_MEASURED = True
    assert lc._compute_grid() == "11x10"

    # An older tt_bio with no flag at all must not stamp either; the census is run against
    # whatever engine the fold imported.
    del stub.COMPUTE_GRID_MEASURED
    assert lc._compute_grid() is None


def test_a_device_already_on_the_guessed_grid_still_counts_as_measured():
    """_configure_active_compute_grid returns early when the device presents the grid the
    module already holds, which is every p300c. A flag set after that return would mark the
    one case it exists to cover as unmeasured."""
    src = pathlib.Path(WT, "tt_bio", "tenstorrent.py").read_text()
    body = src.split("def _configure_active_compute_grid(")[1]
    body = body[:body.index("\ndef ")]
    first_return = min(i for i, line in enumerate(body.splitlines())
                       if line.strip() == "return")
    set_at = min(i for i, line in enumerate(body.splitlines())
                 if line.strip() == "COMPUTE_GRID_MEASURED = True")
    assert set_at < first_return, \
        "COMPUTE_GRID_MEASURED is set past an early return in _configure_active_compute_grid"


def test_a_stats_dict_row_reads_its_own_keys(monkeypatch):
    """`FP32_SOFTMAX_STATS` carries several levers' counters in one dict, and the stats-dict
    reader hardcoded `calls`/`blocked` -- the bias hoist's pair. Any other row built on that dict
    would have reported the bias hoist's numbers as its own and read healthy whether or not it
    ever ran, which is the whole question the census exists to answer.
    """
    lc = _lever_census()
    stub = types.ModuleType("tt_bio.tenstorrent")
    stub.FP32_SOFTMAX_BIAS_HOIST = True
    stub._FP32_SOFTMAX_L1_GRID = (9, 8)
    # every value distinct, so a row reading the wrong key cannot pass by coincidence
    stub.FP32_SOFTMAX_STATS = {"calls": 1320, "blocked": 440, "l1_blocks": 77,
                               "l1_refused": 3, "l1_cores": 72}
    monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", stub)

    rows = lc._snapshot_process()
    hoist, l1 = rows["FP32_SOFTMAX_BIAS_HOIST"], rows["FP32_SOFTMAX_L1_GRID"]
    assert (hoist["served"], hoist["declined"]) == (1320, 440)
    assert (l1["served"], l1["declined"]) == (77, 3)
    assert l1["resolved"] == "(9, 8)"
    assert l1["gauges"] == {"l1_cores": 72}

    # Negative control: the L1 path dark, the bias hoist's counters untouched. Under the old
    # reader both rows read 1320/440 here and the dark one was indistinguishable from the live
    # one. `l1_cores` drops out entirely rather than reporting 0, which is not a core count.
    stub._FP32_SOFTMAX_L1_GRID = (8, 8)
    stub.FP32_SOFTMAX_STATS.update(l1_blocks=0, l1_refused=0, l1_cores=0)
    rows = lc._snapshot_process()
    assert rows["FP32_SOFTMAX_BIAS_HOIST"]["served"] == 1320
    assert rows["FP32_SOFTMAX_L1_GRID"]["served"] == 0
    assert rows["FP32_SOFTMAX_L1_GRID"]["gauges"] is None


def test_every_stats_dict_row_names_its_counter_keys():
    """The syntax only helps if no row can forget it: a `stats-dict` row without keys falls
    back to nothing and reports None, which is the loud failure this replaced the silent one
    with -- but the table is where it should be caught."""
    lc = _lever_census()
    for flag, _mod, _attr, counter, how in lc.LEVERS:
        if how != "stats-dict":
            continue
        assert counter and ":" in counter, flag + ": stats-dict row names no counter keys"
        served, _, declined = counter.split(":", 1)[1].partition(",")
        assert served and declined, flag + ": stats-dict row needs served,declined keys"


def test_a_gauge_is_never_summed_across_processes():
    """`l1_cores` is assigned, not incremented, so the served/declined sum is the wrong merge
    for it: two workers each on 72 cores would report 144, a grid that does not exist. The
    gauges are unioned like `resolved` is, and a disagreement between processes stays visible
    instead of averaging into a plausible wrong number.
    """
    lc = _lever_census()
    d = pathlib.Path(tempfile.mkdtemp())
    for i, cores in enumerate((72, 72, 64)):
        (d / f"pid{i}.json").write_text(json.dumps({"pid": i, "argv": [], "rows": {
            "FP32_SOFTMAX_L1_GRID": {"resolved": "(9, 8)", "served": 10, "declined": 1,
                                     "rejects": None, "gauges": {"l1_cores": cores}}}}))
    row = [r for r in lc.collect(d, "t", [], 0)["rows"]
           if r["flag"] == "FP32_SOFTMAX_L1_GRID"][0]
    assert row["served"] == 30 and row["declined"] == 3
    assert row["gauges"] == {"l1_cores": "64/72"}


def test_resolved_ignores_a_process_that_never_opened_a_chip():
    """`_apply_grid_thresholds` retunes several resolved defaults at device open, so the
    launcher -- which imports the module and never folds -- still holds the pre-device value.
    Unioned in, the fp32-softmax rectangle reads "(8, 8)/(9, 8)" for a lever that resolved to
    one rectangle everywhere it actually ran: the `11x10/13x10` grid-stamp false alarm again.
    Measured on card 3 with TT_BIO_FORCE_GRID=8,9, where the fold's two processes disagree
    exactly like this.
    """
    lc = _lever_census()
    d = pathlib.Path(tempfile.mkdtemp())
    for i, (grid, res) in enumerate(((None, "(8, 8)"), ("8x9", "(9, 8)"))):
        (d / f"pid{i}.json").write_text(json.dumps({"pid": i, "argv": [], "grid": grid, "rows": {
            "FP32_SOFTMAX_L1_GRID": {"resolved": res, "served": 12, "declined": 0,
                                     "rejects": None, "gauges": None}}}))
    row = [r for r in lc.collect(d, "t", [], 0)["rows"]
           if r["flag"] == "FP32_SOFTMAX_L1_GRID"][0]
    assert row["resolved"] == "(9, 8)"
    # the counters still come from every process, measured or not
    assert row["served"] == 24

    # No process opened a chip: there is nothing to prefer, so say what was read rather than
    # dropping the row to a blank.
    (d / "pid1.json").write_text(json.dumps({"pid": 1, "argv": [], "grid": None, "rows": {
        "FP32_SOFTMAX_L1_GRID": {"resolved": "(8, 8)", "served": 0, "declined": 0,
                                 "rejects": None, "gauges": None}}}))
    row = [r for r in lc.collect(d, "t", [], 0)["rows"]
           if r["flag"] == "FP32_SOFTMAX_L1_GRID"][0]
    assert row["resolved"] == "(8, 8)"


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)

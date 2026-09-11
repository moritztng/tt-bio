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


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)

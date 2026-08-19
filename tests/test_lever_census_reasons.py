"""The census reject-reason emit, tested without a device.

A lever that reads served=0 is either dark or correctly declining, and only the reason tells
them apart. These are the four cells the 2026-08-19 size recheck turned on: the tri-bias
projection (8,1) that no _MM_BLOCK entry covers, openfold3 at c_z=128 that the kt==8 scope
refuses, an in-scope protenix pair projection, and the cross-process reason merge.
"""
import json, sys, types, tempfile, pathlib
WT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, WT + "/scripts")

# a stand-in ttnn so the module imports without hardware
tt = types.ModuleType("ttnn")
tt.bfloat16 = "bf16"; tt.float32 = "fp32"
tt.TILE_LAYOUT = "TILE"; tt.ROW_MAJOR_LAYOUT = "RM"
class BT: DRAM = "DRAM"; L1 = "L1"
tt.BufferType = BT
sys.modules["ttnn"] = tt

import lever_census as LC

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

ok = True
def check(label, got, want):
    global ok
    good = got == want
    ok = ok and good
    print(("PASS " if good else "FAIL ") + label + ": " + repr(got) + (" != " + repr(want) if not good else ""))

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
check("flag off", LC._pp_reason(dict_to := type("t", (T,), {"_PAIR_PROJ_MM": False}),
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

print("\nALL OK" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)

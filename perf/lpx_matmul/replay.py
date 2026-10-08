"""Rebuild a captured ttnn matmul call (bench_wh.py's signature key) as op, args, kwargs."""
import re
import ttnn

TSPEC = re.compile(r"\('T', \(([\d, ]*)\), 'DataType\.(\w+)', 'Layout\.(\w+)', MemoryConfig\(.*?buffer_type=BufferType::(\w+).*?\)\)")
def parse(key):
    parts = key.split("|"); op = parts[0]; a, kw = [], {}
    for p in parts[1:]:
        name, _, val = p.partition("=")
        m = TSPEC.fullmatch(val)
        if m:
            v = ("T", tuple(int(x) for x in m.group(1).split(",") if x.strip()), m.group(2), m.group(4))
        elif name in ("memory_config",):
            v = ttnn.L1_MEMORY_CONFIG if "BufferType::L1" in val else ttnn.DRAM_MEMORY_CONFIG
        elif name == "dtype":
            v = getattr(ttnn, {"BFLOAT16": "bfloat16", "BFLOAT8_B": "bfloat8_b", "BFLOAT4_B": "bfloat4_b",
                               "FLOAT32": "float32"}[val.split(".")[-1]])
        elif name == "core_grid":
            x, y = re.findall(r"\d+", val); v = ttnn.CoreGrid(x=int(x), y=int(y))
        elif name == "program_config":
            v = parse_pc(val)
        elif name == "compute_kernel_config":
            continue                                      # taken from r1's sweep event (logged values)
        elif val in ("True", "False", "None"):
            v = {"True": True, "False": False, "None": None}[val]
        else:
            v = val.strip("'")
        if name.startswith("a") and name[1:].isdigit():
            a.append(v)
        else:
            kw[name] = v
    return op, a, kw

def parse_pc(val):
    if val == "None":
        return None
    cls, body = val.split("(", 1)
    d = {}
    for k, v in re.findall(r"(\w+)=([^,()]+|\(x=\d+,y=\d+\)|\{\})", body):
        if k == "compute_with_storage_grid_size":
            x, y = re.findall(r"\d+", v); d[k] = (int(x), int(y))
        elif k in ("fused_activation", "hop_cores", "num_global_cb_receivers", "out_block_h", "out_block_w"):
            continue
        else:
            d[k] = bool(int(v)) if k in ("transpose_mcast", "fuse_batch", "mcast_in0", "gather_in0", "untilize_out") else int(v)
    d.pop("gather_in0", None); d.pop("untilize_out", None)
    return getattr(ttnn, cls)(**d)

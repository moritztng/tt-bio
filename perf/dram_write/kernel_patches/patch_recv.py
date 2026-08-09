import shutil, os, re
F = "/home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp"
B = F + ".dws_backup"
if not os.path.exists(B):
    shutil.copy2(F, B)
lines = open(B).read().split("\n")

def ind(n):
    return re.match(r"\s*", lines[n - 1]).group(0)

# sanity: the lines we are about to wrap must be what we think they are
assert "for (uint32_t sbh" in lines[163], lines[163]
assert "cb_out.wait_front(out_subblock_tile_count);" in lines[179], lines[179]
assert "noc.async_write_barrier();" in lines[203], lines[203]
assert "Pop row(s)" in lines[215], lines[215]

new = []
for n, l in enumerate(lines, 1):
    if n == 164:
        new.append(ind(164) + "{")
        new.append(ind(164) + 'DeviceZoneScopedN("OUT_SECTION");')
        new.append(l)
    elif n == 180:
        i = ind(180)
        new += [i + "{", i + '    DeviceZoneScopedN("SB_WAIT");', i + "    " + l.strip(), i + "}"]
    elif n == 204:
        i = ind(204)
        new += [i + "{", i + '    DeviceZoneScopedN("SB_DRAIN");', i + "    " + l.strip(), i + "}"]
    elif n == 216:
        new.append(ind(216) + "}")
        new.append(l)
    else:
        new.append(l)
open(F, "w").write("\n".join(new))
print("PATCHED receiver ok")

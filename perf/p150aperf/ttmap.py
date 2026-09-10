import os, glob, ttnn
d = ttnn.open_device(device_id=0)
nodes = sorted(set(os.path.realpath(p) for p in glob.glob("/proc/self/fd/*") if "tenstorrent" in (os.path.realpath(p) if os.path.exists(p) else "")))
print("TTVD", os.environ.get("TT_VISIBLE_DEVICES"), "NUM", ttnn.GetNumAvailableDevices(), "NODES", nodes)
ttnn.close_device(d)

"""Probe the target chip: grid, L1, arch, ttnn version. No model code."""
import os, json, sys
import ttnn

dev = ttnn.open_device(device_id=int(os.environ.get("PROBE_DEV", "0")))
try:
    g = dev.compute_with_storage_grid_size()
    info = {
        "ttnn_version": getattr(ttnn, "__version__", "?"),
        "arch": str(dev.arch()),
        "grid_x": g.x, "grid_y": g.y, "cores": g.x * g.y,
        "l1_size_per_core": getattr(dev, "l1_size_per_core", lambda: None)(),
        "dram_grid": str(dev.dram_grid_size()),
        "num_devices_visible": ttnn.GetNumAvailableDevices(),
    }
    print(json.dumps(info, indent=2))
finally:
    ttnn.close_device(dev)

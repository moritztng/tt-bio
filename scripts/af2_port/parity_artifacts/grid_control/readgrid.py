"""Read the active tt-bio compute grid on this box, through tt-bio's own device path.

A bare ttnn.open_device dies on qb2 without an MGD; tt_bio.get_device hands a 1-chip worker
its own 1x1 descriptor, so this is the only way to see the grid the engine actually picks.
"""
import json
import os

from tt_bio import tenstorrent as T

dev = T.get_device()
a = dev.compute_with_storage_grid_size()
print("GRIDREPORT " + json.dumps({
    "force_grid_env": os.environ.get("TT_BIO_FORCE_GRID"),
    "device_reports": [int(a.x), int(a.y)],
    "COMPUTE_GRID_MAIN": list(T.COMPUTE_GRID_MAIN),
    "CORE_GRID_MAIN": [T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y],
    "COMPUTE_GRID_X_13": T.COMPUTE_GRID_X_13,
    "COMPUTE_GRID_X_11": T.COMPUTE_GRID_X_11,
    "COMPUTE_GRID_Y": T.COMPUTE_GRID_Y,
}))

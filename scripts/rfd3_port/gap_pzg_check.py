#!/usr/bin/env python3
"""Why does RFD3_PZ_GATHER move the design digest when gap_pz_embed says gather_tiled == embedding + to_layout?

    gap_pzg_check.py OUT.json [STEPS]

Runs one fast design on the page fixture with RFD3_PZ_GATHER on and, on every _process_z_collapsed call, also
runs the shipped embedding route on the same inputs and compares the gathered table rows (logical and padded)
and the returned z. Records per call: table rows N, batch, max |diff| and the count of unequal elements.
"""
import json
import os
import pathlib
import sys

sys.argv = [sys.argv[0], "pws+at+apb+dpb+css+pwr+qkh+gch+asdpa+dsdpa+sf+rs+kd+pzg", "/tmp/gap_pzg_check",
            sys.argv[2] if len(sys.argv) > 2 else "4", "60", "--fast=1", "OUT=" + sys.argv[1]]
OUT = pathlib.Path(sys.argv.pop()[4:])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_knobs import apply                                               # noqa: E402
apply(sys.argv[1])
sys.path.insert(0, os.getcwd())
import torch                                                              # noqa: E402
import ttnn                                                               # noqa: E402
from tt_bio import gather_tiled as G                                      # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402

REC = []
_orig = M.DiffusionTokenEncoder._process_z_collapsed


def _check(self, Z_init_II, bins, bins_self, batch, I):
    dev, n = self.device, self.N_BINS
    tab = self._process_z_table(bins_self is not None)
    flat = bins if bins_self is None else bins * n + bins_self
    idx_a = M._tt_idx(flat, dev)
    em_a = ttnn.embedding(idx_a, tab, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    em_a = ttnn.to_layout(ttnn.reshape(em_a, (batch, I, I, self.C_Z)), ttnn.TILE_LAYOUT)
    rm = ttnn.to_layout(tab, ttnn.ROW_MAJOR_LAYOUT)
    idx_g = ttnn.from_torch(G.pad_index(flat, batch * I, I), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                            dtype=ttnn.uint32)
    em_g = G.gather_tiled(idx_g, rm, batch, I)
    a, g = ttnn.to_torch(em_a).float(), ttnn.to_torch(em_g).float()
    href = ttnn.to_torch(tab).float()[flat.reshape(-1).long()].reshape(a.shape)
    M._PZ_GATHER = False
    z_a = ttnn.to_torch(_orig(self, Z_init_II, bins, bins_self, batch, I)).float()
    M._PZ_GATHER = True
    z_g_t = _orig(self, Z_init_II, bins, bins_self, batch, I)
    z_g = ttnn.to_torch(z_g_t).float()
    r = {"call": len(REC), "N": int(tab.shape[0]), "tab_shape": list(tab.shape), "tab_dtype": str(tab.dtype),
         "batch": batch, "I": I, "flat_dtype": str(flat.dtype), "flat_shape": list(flat.shape),
         "flat_max": int(flat.max()), "em_shape": list(a.shape),
         "em_ship_vs_host": float((a - href).abs().max()), "em_gather_vs_host": float((g - href).abs().max()),
         "em_neq": int((a != g).sum()), "z_maxabs": float((z_a - z_g).abs().max()), "z_neq": int((z_a != z_g).sum())}
    print("[pzg]", r, flush=True)
    REC.append(r)
    OUT.write_text(json.dumps(REC, indent=1))
    return z_g_t


M.DiffusionTokenEncoder._process_z_collapsed = _check
import runpy                                                              # noqa: E402
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_screen.py"), run_name="__main__")

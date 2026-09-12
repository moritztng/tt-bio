"""Probe: what the installed ttnn exposes for mesh + collectives, and what the box's mesh looks like."""
import os, sys, inspect
import ttnn

print("=== fabric enum ===")
print([x for x in dir(ttnn.FabricConfig) if not x.startswith("_")])
print("=== all_gather sig ===")
for name in ("all_gather", "reduce_scatter", "all_to_all_dispatch", "mesh_partition"):
    f = getattr(ttnn, name, None)
    print(name, "->", getattr(f, "__doc__", "")[:1200] if f is not None else "MISSING")
print("=== experimental ccl ===")
exp = getattr(ttnn, "experimental", None)
if exp is not None:
    print(sorted(n for n in dir(exp) if any(k in n.lower() for k in ("all_gather", "reduce_scatter", "all_to_all", "ccl", "send", "recv"))))
print("=== system mesh ===")
try:
    print(ttnn.visualize_system_mesh())
except Exception as e:
    print("visualize_system_mesh failed:", e)

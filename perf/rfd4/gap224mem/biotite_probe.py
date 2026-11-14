"""Memory of biotite's CCD access paths: indexed get_from_ccd vs a full-column decode."""
import time
def m(t):
    d={l.split()[0]:int(l.split()[1])>>10 for l in open("/proc/self/smaps_rollup") if l.startswith(("Rss","Pss:"))}; print(t,d,flush=True)
import biotite.structure.info.ccd as c
m("import")
t=time.time(); b=c.get_ccd(); m(f"get_ccd {time.time()-t:.2f}")
t=time.time(); r=c.get_from_ccd("chem_comp_atom","ALA"); m(f"from_ccd ALA {time.time()-t:.2f}"); print(type(r))
t=time.time(); [c.get_from_ccd(k,"ALA") for k in ("chem_comp","chem_comp_bond")]; m(f"others {time.time()-t:.2f}")
t=time.time(); [c.get_from_ccd(k,x) for k in ("chem_comp","chem_comp_atom","chem_comp_bond") for x in ("TRP","GLY","HEM")]; m(f"3 more {time.time()-t:.2f}")
t=time.time(); a=b["chem_comp_atom"]["comp_id"].as_array(); m(f"one full column decode {time.time()-t:.2f} {a.dtype} {a.shape}")

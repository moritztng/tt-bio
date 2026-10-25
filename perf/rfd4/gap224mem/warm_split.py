import json, sys, time
sys.path.insert(0, sys.argv[1])
def mem(tag, t0):
    d = {l.split()[0][:-1]: int(l.split()[1]) // 1024 for l in open("/proc/self/smaps_rollup") if l.startswith(("Rss", "Pss"))}
    print(json.dumps({"stage": tag, **d, "s": round(time.time() - t0, 2), "mods": len(sys.modules)}), flush=True)
t = time.time(); import torch; mem("torch", t)
t = time.time(); from tt_bio._vendor.rfd4_prep._cif import transforms_unindexing; mem("transforms_unindexing", t)
t = time.time(); from tt_bio._vendor.atomworks.io.template import get_empty_ccd_template; mem("template import", t)
t = time.time(); get_empty_ccd_template("ALA"); mem("ALA", t)
t = time.time(); [get_empty_ccd_template(c) for c in ("ARG","GLY","TRP")]; mem("3 more", t)

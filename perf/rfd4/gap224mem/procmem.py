"""PSS / RSS / private memory of a process and its tree, from /proc."""
import os


def rollup(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            k, *v = line.split()
            if k in ("Pss:", "Rss:", "Private_Clean:", "Private_Dirty:"):
                d[k[:-1]] = int(v[0]) / 1024
    except OSError:
        return None
    d["Private"] = d.pop("Private_Clean", 0) + d.pop("Private_Dirty", 0)
    return d


def tree(root):
    kids, out = {}, [root]
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                pp = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            except OSError:
                continue
            kids.setdefault(pp, []).append(int(p))
    for p in out:
        out += kids.get(p, [])
    return out

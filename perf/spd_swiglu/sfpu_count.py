import re,sys
# count instructions per function in a .s file; split SFPU (sfp*/TT sfpu) vs other
def funcs(path):
    cur=None; out={}
    for line in open(path):
        m=re.match(r'^([A-Za-z_][\w.$]*):', line)
        if m and not m.group(1).startswith('.'): cur=m.group(1); out[cur]=[]; continue
        s=line.strip()
        if cur and s and not s.startswith('.') and not s.endswith(':') and not s.startswith('#'):
            out[cur].append(s.split()[0])
    return out
for p in sys.argv[1:]:
    for f,ins in funcs(p).items():
        sf=[i for i in ins if i.lower().startswith('sfp') or i.startswith('tt')]
        from collections import Counter
        c=Counter(sf)
        print(p, f, 'total', len(ins), 'sfpu', len(sf), 'nop', c.get('sfpnop',0)+c.get('ttnop',0), dict(c.most_common(8)))

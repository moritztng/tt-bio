import re,sys,collections
KEYS=["expectation","expectation_1","expectation_2","expectation_6","maximization","reconstruction",
      "flatten solvent","update resolution","iterate:  writeOutput"," RcT1_BPrefRecon",
      "expectationSomeParticles","mpiWaitEndOfExpectation","mpiCombineThroughNetwork"]
for arm in sys.argv[1:]:
    txt=open("/home/ttuser/relion-scratch/p1/%s.log"%arm,errors="ignore").read()
    # split into timing tables: each table starts after a line of "="*N following "Expectation iteration"
    tables=[]; cur=None
    for line in txt.replace("\r","\n").split("\n"):
        m=re.match(r"^(.{0,40}?)\s*:\s*([0-9.]+) sec \((\d+) microsec/operation\)",line)
        if m:
            name=m.group(1).rstrip()
            if name=="expectation" and cur is not None: tables.append(cur); cur=None
            if cur is None: cur={}
            cur[name]=(float(m.group(2)),int(m.group(3)))
    if cur: tables.append(cur)
    print("### arm %s : %d timing tables"%(arm,len(tables)))
    for i,t in enumerate(tables):
        print("  table %d: "%i + "  ".join("%s=%.3f"%(k,t[k][0]) for k in KEYS if k in t))

import sys
variant=sys.argv[1]; arm=sys.argv[2]; steps=int(sys.argv[3])
sys.argv=["x","--out","/tmp/x.json","--n","384","--arms",arm,"--warm","0","--pairs",str(steps)]
import ttnn, torch
real=ttnn.matmul
stat={"calls":0,"first_bad":None,"bad":0}
def mm(a,b,*args,**kw):
    pc=kw.get("program_config")
    if len(a.shape)==4 and pc is not None:
        if kw.get("transpose_b"):
            if variant=="tb_none": kw["program_config"]=None
            if variant=="explicit_t":
                kw["transpose_b"]=False; b=ttnn.transpose(b,-2,-1)
        fin_in = bool(ttnn.to_torch(a).float().isfinite().all()) and bool(ttnn.to_torch(b).float().isfinite().all())
        y=real(a,b,*args,**kw)
        stat["calls"]+=1
        if fin_in and not bool(ttnn.to_torch(y).float().isfinite().all()):
            stat["bad"]+=1
            if stat["first_bad"] is None: stat["first_bad"]=(tuple(a.shape),kw.get("transpose_a"),kw.get("transpose_b"),kw.get("program_config") is not None)
        return y
    return real(a,b,*args,**kw)
ttnn.matmul=mm
import perf.bcx_triatt.block_ab as b
import perf.hallgrad.census as c
orig=c.one_step
def safe(*a,**k):
    try: return orig(*a,**k)
    except RuntimeError: return {"step_s":0,"fwd_s":0,"bwd_s":0,"reached":0}
b.one_step=safe
b.main()
print("VARIANT",variant,arm,stat)

import numpy as np
def bf16(v): 
    b=np.float32(v).view(np.uint32); b=(b+0x7fff+((b>>16)&1))&0xffff0000; return float(np.uint32(b).view(np.float32))
def f32(v): return float(np.float32(v))
def relfit_fixed(f, lo, hi, deg, fixed, n=3000, iters=40):
    """fixed: dict power->value. Fit the rest near-minimax in relative error."""
    x = np.cos(np.linspace(0, np.pi, n))*(hi-lo)/2+(hi+lo)/2
    y=f(x); free=[k for k in range(deg+1) if k not in fixed]
    base=sum(v*x**k for k,v in fixed.items()) if fixed else 0*x
    V=np.stack([x**k for k in free],1); w=np.ones_like(x)
    for _ in range(iters):
        A=V/y[:,None]*np.sqrt(w)[:,None]; b=(1-base/y)*np.sqrt(w)
        c,*_=np.linalg.lstsq(A,b,rcond=None)
        e=np.abs((V@c+base)/y-1); w=w*(e+1e-300); w/=w.sum()
    coef=dict(fixed); coef.update(dict(zip(free,c))); return [coef[k] for k in range(deg+1)]
def horner32(c, x):
    x=x.astype(np.float32); p=np.float32(c[-1])*np.ones_like(x)
    for ck in c[-2::-1]: p=(p*x+np.float32(ck)).astype(np.float32)
    return p
def err(c,f,lo,hi):
    xx=np.linspace(lo,hi,400001); return np.max(np.abs(horner32(c,xx).astype(np.float64)/f(xx)-1))
f=lambda r: 2.0**r
for deg,nb in ((4,1),(4,2),(5,1),(5,2),(5,3)):
    # round the top nb coefficients to bf16, refit the rest, rounding to fp32
    fixed={}
    c=relfit_fixed(f,-0.5,0.5,deg,fixed)
    for k in range(deg,deg-nb,-1):
        fixed[k]=bf16(c[k]); c=relfit_fixed(f,-0.5,0.5,deg,fixed)
    c=[f32(v) if k not in fixed else v for k,v in enumerate(c)]
    print(f"exp deg{deg} bf16-top{nb}: err {err(c,f,-0.5,0.5):.2e}", ["%.9g"%v for v in c])
g=lambda e: 1/(1+e)
for deg in (1,2,3):
    c=relfit_fixed(g,0,1,deg,{}); cb=[bf16(v) for v in c]
    e0=err(cb,g,0,1); print(f"recip seed deg{deg} bf16: err {e0:.2e} 1N {e0**2:.1e} 2N {e0**4:.1e}", cb)
print("---- c0=1 fixed")
for nb in (2,):
    fixed={0:1.0}; c=relfit_fixed(f,-0.5,0.5,4,fixed)
    for k in range(4,4-nb,-1):
        fixed[k]=bf16(c[k]); c=relfit_fixed(f,-0.5,0.5,4,fixed)
    c=[f32(v) if k not in fixed else v for k,v in enumerate(c)]
    print(f"exp deg4 c0=1 bf16-top{nb}: err {err(c,f,-0.5,0.5):.2e}", ["%.9g"%v for v in c])
    fixed={0:1.0}; c=relfit_fixed(f,-0.5,0.5,5,fixed)
    for k in range(5,2,-1):
        fixed[k]=bf16(c[k]); c=relfit_fixed(f,-0.5,0.5,5,fixed)
    c=[f32(v) if k not in fixed else v for k,v in enumerate(c)]
    print(f"exp deg5 c0=1 bf16-top3: err {err(c,f,-0.5,0.5):.2e}", ["%.9g"%v for v in c])

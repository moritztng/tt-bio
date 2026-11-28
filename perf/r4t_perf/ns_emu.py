import torch
A,B,C,EPS=3.4445,-4.775,2.0315,1e-7
bf=lambda t: t.to(torch.bfloat16).double()
def ref(x,dt):
    x=x.to(dt); x=x/x.norm(dim=(-2,-1),keepdim=True).clamp(min=EPS)
    for _ in range(5):
        g=torch.bmm(x,x.transpose(-1,-2)); x=torch.baddbmm(x,torch.baddbmm(g,g,g,beta=B,alpha=C),x,beta=A)
    return x.double()
def emu(x,mode):
    x=bf(x.double()); x=bf(x/x.norm(dim=(-2,-1),keepdim=True))
    for _ in range(5):
        g=x@x.transpose(-1,-2)
        if mode=='bf16':
            g=bf(g); gg=bf(g@g); p=bf(bf(g*B)+bf(gg*C)); x=bf(bf(x*A)+bf(p@x))
        elif mode=='f32':   # current device f32 arm: A*x rounded to bf16 before the add
            gb=bf(g); p=g*B+(gb@gb)*C; x=bf(bf(x*A)+(bf(p)@x))
        elif mode=='fused': # one rounding per baddbmm, like host
            gb=bf(g); p=bf(g*B+(gb@gb)*C); x=bf(x*A+p@x)
        elif mode=='host':
            g=bf(g); p=bf(g*B+(g@g)*C); x=bf(x*A+p@x)
        elif mode=='gb32':  # consistent rounded gram, p fp32 then bf16 input, A*x fp32
            g=bf(g); p=g*B+(g@g)*C; x=bf(x*A+bf(p)@x)
        elif mode=='gbdev': # device-feasible: gb used in both terms, A*x rounded separately (current dev add order)
            g=bf(g); p=g*B+(g@g)*C; x=bf(bf(x*A)+bf(p)@x)
        elif mode=='x32':   # keep x fp32 between iterations, matmul inputs bf16
            xb=bf(x); g=xb@xb.transpose(-1,-2); gb=bf(g); p=g*B+(gb@gb)*C; x=x*A+bf(p)@xb
    return bf(x)
torch.manual_seed(0)
w=torch.randn(8,256,768)
r=ref(w,torch.float64); h=ref(w,torch.bfloat16)
e=lambda o:((o-r).flatten(1).norm(dim=1)/r.flatten(1).norm(dim=1))
print('host',e(h).max().item())
for m in ['bf16','f32','fused','x32','host','gb32','gbdev']: print(m,e(emu(w,m)).max().item(), (e(emu(w,m))/e(h)).max().item())

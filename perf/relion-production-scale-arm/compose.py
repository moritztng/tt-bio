N1=4452.0
G1={4452:39.00,13356:74.21,35616:179.59,111300:548.21}
G2={111300:310.65}
CX={4452:71.58,13356:179.13,35616:451.41,111300:1369.08}
QB1={4452:(186.48,176.344),13356:(662.39,643.288),35616:(1772.23,1737.06)}
CS={4452:[.9052,.9083,.9073],13356:[.9100,.9064,.9080],35616:[.9076,.9095,.9073]}
CS17=[.7613,.7615,.7610,.7627]
def m(v):return sum(v)/len(v)
def fit(d):
    xs=sorted(d);ys=[d[x] for x in xs];n=len(xs);sx=sum(xs);sy=sum(ys)
    sxx=sum(x*x for x in xs);sxy=sum(x*y for x,y in zip(xs,ys))
    mm=(n*sxy-sx*sy)/(n*sxx-sx*sx);return (sy-mm*sx)/n, mm*1000
def fan(k):return 2370.0*k/(117.4+74.1*k)
print("fanout k=1 %.2fx  k=25 %.2fx"%(fan(1),fan(25)))
print("it13 coarse share by N:", {k:round(100*m(v),2) for k,v in CS.items()})
print("it17 coarse share      : %.2f%% (n=4 ranks, spread %.2f pt)"%(100*m(CS17),100*(max(CS17)-min(CS17))))
for n,d in (("Xeon",CX),("H200",G1)):
    F,mm=fit(d);xs=sorted(d)
    mg=[(d[b]-d[a])/(b-a)*1000 for a,b in zip(xs,xs[1:])]
    print("%s fit F=%.2fs m=%.3f ms/p  adjacent %s"%(n,F,mm,["%.3f"%x for x in mg]))
print("1->2 H200 @111300 %.3fx"%(G1[111300]/G2[111300]))

# --- ONE ITERATION 13 composition, Xeon-class host
e6w={k:QB1[k][1]/QB1[k][0] for k in QB1}
print("exp6/wall:",{k:round(100*v,2) for k,v in e6w.items()})
DEV=9.37
rows={}
for N,sw in ((4452,m(CS[4452])*e6w[4452]),(111300,m([x for v in CS.values() for x in v])*e6w[35616])):
    k=N/N1;wall=CX[N];res=wall*(1-sw);dev=DEV*k;f=fan(k)
    rows[N]=(res,dev,f,res+dev,res+dev/f)
    print("N=%d coarse-of-wall %.2f%% host-residue %.2f s dev1 %.2f s | 1xp150 %.2f | GALAXY %.2f | H200 %.2f (%.2fx)"%(
        N,100*sw,res,dev,res+dev,res+dev/f,G1[N],G1[N]/(res+dev/f)))
print("  GALAXY vs 2xH200 @111300: %.2fx"%(G2[111300]/rows[111300][4]))

# --- WHOLE REFINEMENT, trajectory coarse share 78.2% (e2e §4 MEASURED), loop=94.5% of wall
TR=0.782; LOOPW=0.945
xw=527.12; XF,Xm1=fit(CX)
Xm_whole=(xw-XF)/N1*1000            # ms/p, Xeon whole refinement
nc=Xm_whole*(1-TR*LOOPW)            # non-coarse host marginal
devw=46.8/N1*1000                   # ms/p, 5 cs=196 iterations on one p150
print("\nWHOLE REFINEMENT  Xeon m=%.2f ms/p  coarse-of-wall %.1f%%  non-coarse host %.2f ms/p  dev %.3f ms/p"%(
    Xm_whole,100*TR*LOOPW,nc,devw))
for N in (4452,111300):
    k=N/N1;g=XF+(nc+devw/fan(k))*N/1000;p=XF+(nc+devw)*N/1000
    print("  N=%7d  GALAXY %9.1f s   1xp150 %9.1f s"%(N,g,p))
gal25=XF+(nc+devw/fan(25))*111300/1000
# H200 whole refinement at 111300: two independent routes
h_lo=12.57+((129.89-12.57)/N1*1000)*1.232*111300/1000     # own marginal x measured 23% degradation
h_hi=(XF+Xm_whole*111300/1000)/(CX[111300]/G1[111300])    # composed Xeon whole / measured 1-iter ratio
print("  1xH200 whole @111300: %.0f s (own marginal x degradation) .. %.0f s (Xeon/ratio route)"%(h_lo,h_hi))
print("  GALAXY %.0f s  =>  vs 1xH200 %.2fx .. %.2fx ; vs 2xH200 %.2fx .. %.2fx (1->2 = 1.765x)"%(
    gal25,h_lo/gal25,h_hi/gal25,h_lo/1.765/gal25,h_hi/1.765/gal25))
# energy, whole refinement, e2e convention: accelerator powered for the whole wall
E_G=4.32*gal25/3600; E_H=0.1308*h_lo/3600; E_H2=0.1308*2*(h_lo/1.765)/3600
print("  energy: GALAXY %.3f kWh | 1xH200 %.4f kWh | 2xH200 %.4f kWh => %.1fx / %.1fx"%(
    E_G,E_H,E_H2,E_G/E_H,E_G/E_H2))
print("  (at 4452 it was 0.1698 vs 0.0048 = 35.4x)")

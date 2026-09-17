import time
p="/home/ttuser/qbcard/cardtel.tsv"
hdr=None; rows=[]
for line in open(p,errors="replace"):
    f=line.rstrip("\n").split("\t")
    if f[0].startswith("#"):
        if len(f)>20: hdr=[f[0][1:]]+f[1:]
        continue
    if hdr and len(f)==len(hdr): rows.append(f)
i={n:k for k,n in enumerate(hdr)}
def g(r,n):
    try: return float(r[i[n]])
    except: return None
last=float(rows[-1][0]); print("file ends", time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime(last)),
                                "| spans %.1f h"%((last-float(rows[0][0]))/3600))
# ABSOLUTE windows, today only
WIN=[("768aa fold->wedge",1789606320,1789607700),("census smoke",1789608400,int(last))]
for name,a,b in WIN:
    print(f"\n=== {name} ===")
    print(f"{'utc':>9s} {'aiclk':>5s} {'pwr_W':>6s} {'mst_rd/s':>10s} {'slv_rd/s':>10s}")
    prev=None
    for r in rows:
        t=g(r,"epoch")
        if t is None or not (a<=t<=b): 
            if t is not None and t<a: prev=r
            continue
        if prev is not None:
            dt=g(r,"epoch")-g(prev,"epoch")
            if dt>0 and int(t)%60<6:
                d1=(g(r,'c0_mst_rd_data_word_received0')-g(prev,'c0_mst_rd_data_word_received0'))/dt
                d2=(g(r,'c0_slv_rd_data_word_sent0')-g(prev,'c0_slv_rd_data_word_sent0'))/dt
                pw=g(r,'c0_power1') or 0
                print(f"{time.strftime('%H:%M:%S',time.gmtime(t)):>9s} {g(r,'c0_tt_aiclk') or 0:5.0f} "
                      f"{pw/1e6:6.1f} {d1:10.0f} {d2:10.0f}")
        prev=r

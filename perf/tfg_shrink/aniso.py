import sys, numpy as np
sys.path.insert(0, "/home/moritz/.coworker/wt/tfg-ship-logs/sweep")
from compress_scan import chains
ref = chains(sys.argv[1]); 
for f in sys.argv[2:]:
    c = chains(f)
    for ch in ("A","B"):
        X=np.array(c[ch]); Y=np.array(ref[ch]); n=min(len(X),len(Y)); X=X[:n]-X[:n].mean(0); Y=Y[:n]-Y[:n].mean(0)
        A,res,_,_=np.linalg.lstsq(Y,X,rcond=None)   # X ~ Y A
        print(f.split("_")[-1], ch, "singular values of best linear map", np.round(np.linalg.svd(A,compute_uv=False),3), "Rg", round(np.sqrt((X**2).sum(1).mean()),2), "vs", round(np.sqrt((Y**2).sum(1).mean()),2))
    X=np.vstack([c["A"],c["B"]]); print("  AB combined Rg", round(np.sqrt(((X-X.mean(0))**2).sum(1).mean()),2))

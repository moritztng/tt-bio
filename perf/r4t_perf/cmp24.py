import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
print("fwd", a["fwd_rel"], b["fwd_rel"], "digest", a["output_digest"], b["output_digest"])
g=a["grads"]; h=b["grads"]; bad=0
for k in g:
  ra,ca=g[k]["rel_l2"],g[k]["cos"]; rb,cb=h[k]["rel_l2"],h[k]["cos"]
  lim = ra+1e-4 if ra<1e-3 else 1.10*ra
  ok = rb<=lim and cb>=ca-1e-5; bad+= not ok
  print("%-45s off %.3e on %.3e ratio %.3f cos %.7f %.7f %s"%(k,ra,rb,rb/ra,ca,cb,"ok" if ok else "FAIL"))
print("fails",bad)

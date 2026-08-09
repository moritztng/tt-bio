import json
d = json.load(open("perf/dit_attn/diffops_protenix-v2_after.json"))
r = d["records"]
NT, H, HD, F = 320, 16, 64, 4
pair = H * NT * NT * F
qkv = H * NT * HD * F
READ, WRITE, COMPUTE = 397.5e9, 272.2e9, 93.44e12
us = lambda b, roof: b / roof * 1e6
sites = [("1654", "q@kT",    2 * qkv,     pair, 2 * H * NT * NT * HD),
         ("1656", "add z",   2 * pair,    pair, 0),
         ("1657", "softmax", pair,        pair, 0),
         ("1661", "attn@v",  pair + qkv,  qkv,  2 * H * NT * HD * NT)]
print("pair %.3f MB  qkv %.3f MB" % (pair / 1e6, qkv / 1e6))
print("site   op        meas_us   rd_us   wr_us   floor   %roof  roof    AI")
tm = tf = 0.0
for s, nm, rb, wb, fl in sites:
    recs = [x for x in r if x["site"] == "tenstorrent.py:" + s]
    m = sum(x["s"] for x in recs) / len(recs) * 1e6
    ru, wu = us(rb, READ), us(wb, WRITE)
    floor, roof = (ru, "read") if ru >= wu else (wu, "write")
    ai = fl / (rb + wb) if fl else 0.0
    tm += m; tf += floor
    print("%-6s %-8s %8.2f %7.2f %7.2f %7.2f %6.1f%% %6s %6.1f"
          % (s, nm, m, ru, wu, floor, floor / m * 100, roof, ai))
    if fl:
        rate = fl / (m * 1e-6)
        print("         -> %.2f TFLOP/s = %.1f%% of the %.2f TFLOP/s fp32 compute roof"
              % (rate / 1e12, rate / COMPUTE * 100, COMPUTE / 1e12))
print("CHAIN            %8.2f %7s %7s %7.2f %6.1f%%" % (tm, "", "", tf, tf / tm * 100))
print("per fold (x4800): chain %.0f ms, floor %.0f ms, residual %.0f ms"
      % (tm * 4800 / 1000, tf * 4800 / 1000, (tm - tf) * 4800 / 1000))
for s in ("1614", "1622", "1689", "1690", "1691", "1692"):
    recs = [x for x in r if x["site"] == "tenstorrent.py:" + s]
    if recs:
        print("  %s %-22s %7.2f us/call" % (s, recs[0]["op"],
              sum(x["s"] for x in recs) / len(recs) * 1e6))
print("stage_warm_s", d["stage_warm_s"], "steps_per_fold", d["steps_per_fold"])

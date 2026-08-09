import json
d = json.load(open("perf/dit_attn/diffops_opendde_after.json"))
r = d["records"]
NT, H, HD, F = 608, 16, 64, 2          # bf16
pair, qkv = H * NT * NT * F, H * NT * HD * F
READ, WRITE, COMPUTE = 390.3e9, 272.1e9, 109.82e12   # bf16 roofs, this card
us = lambda b, roof: b / roof * 1e6
sites = [("1654", "q@kT",    2 * qkv,    pair, 2 * H * NT * NT * HD),
         ("1656", "add z",   2 * pair,   pair, 0),
         ("1657", "softmax", pair,       pair, 0),
         ("1661", "attn@v",  pair + qkv, qkv,  2 * H * NT * HD * NT)]
print("pair %.3f MB  qkv %.3f MB   (bf16, NT=%d)" % (pair / 1e6, qkv / 1e6, NT))
print("site   op        meas_us   rd_us   wr_us   floor   %roof  roof     AI")
tm = tf = 0.0
for s, nm, rb, wb, fl in sites:
    recs = [x for x in r if x["site"] == "tenstorrent.py:" + s]
    m = sum(x["s"] for x in recs) / len(recs) * 1e6
    ru, wu = us(rb, READ), us(wb, WRITE)
    floor, roof = (ru, "read") if ru >= wu else (wu, "write")
    tm += m; tf += floor
    print("%-6s %-8s %8.2f %7.2f %7.2f %7.2f %6.1f%% %6s %6.1f"
          % (s, nm, m, ru, wu, floor, floor / m * 100, roof, (fl / (rb + wb)) if fl else 0.0))
    if fl:
        print("         -> %.2f TFLOP/s = %.1f%% of the %.2f TFLOP/s bf16 compute roof"
              % (fl / (m * 1e-6) / 1e12, fl / (m * 1e-6) / COMPUTE * 100, COMPUTE / 1e12))
print("CHAIN            %8.2f %7s %7s %7.2f %6.1f%%" % (tm, "", "", tf, tf / tm * 100))
calls = 24 * d["steps_per_fold"]
print("per fold (x%d): chain %.0f ms, floor %.0f ms, residual %.0f ms"
      % (calls, tm * calls / 1000, tf * calls / 1000, (tm - tf) * calls / 1000))
print("implied baseline chain in-fold from the 814 ms fold delta: %.1f us/call"
      % (tm + 814000.0 / calls))

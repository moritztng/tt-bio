"""D41(2) second condition: per batch, |total - upstream fp32 total| card-packed (R) <= 2 x host-packed (five).
Total = sum of the step-1 loss terms; upstream fp32 total = sum of upstream_terms in the same record.
Usage: r2cmp.py R_DIR R_PREFIX H_DIR H_PREFIX"""
import json, sys
rd, rp, hd, hp = sys.argv[1:5]
def rec(path):
    return [json.loads(x) for x in open(path) if x.startswith('{"step"')][0]
ok = True
for b in range(9):
    r, h = rec(f"{rd}/{rp}_batch{b:02d}.log"), rec(f"{hd}/{hp}_batch{b:02d}.log")
    up = sum(h["upstream_terms"].values())
    assert abs(up - sum(r["upstream_terms"].values())) < 1e-12 * max(1, abs(up)), "different upstream reference"
    dr, dh = abs(sum(r["terms"].values()) - up), abs(sum(h["terms"].values()) - up)
    p = dr <= 2 * dh
    ok &= p
    print(f"batch{b:02d} up {up:.6f} |card-up| {dr:.3e} |host-up| {dh:.3e} ratio {dr/max(dh,1e-300):.2f} {'PASS' if p else 'FAIL'}"
          f" digest card {r['master_digest']} host {h['master_digest']}")
print("R2-DRIFT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

"""Does `widths[i+1:]` keep every literal and the opendde win, and drop only the self-pairs?"""
NEW = {(8,24):(4,8,1,4,1),(8,8):(4,8,1,4,1),(4,12):(4,4,1,4,1),(4,4):(4,4,1,4,1),
       (2,12):(4,2,1,4,1),(2,2):(4,2,1,4,1),(2,6):(4,2,1,4,1),
       (12,36):(4,12,1,2,1),(12,12):(8,12,1,2,1)}
LITERALS = {(4,16):(4,4,1,4,1),(4,17):(4,4,1,4,1),(8,32):(4,8,1,4,1),(8,33):(4,8,1,4,1),
            (2,8):(4,2,1,4,1),(2,9):(4,2,1,4,1)}

def fused(kt, nt, strict):
    widths = sorted({n for (k, n) in NEW if k == kt})
    best = None
    for i, a in enumerate(widths):
        for b in (widths[i+1:] if strict else widths[i:]):
            if nt in (a+b, a+b+1) and (best is None or max(a,b) > best):
                best = max(a, b)
    return NEW[(kt, best)] if best is not None else None

print("with widths[i+1:]  (the one-character fix):")
ok = all(fused(*k, True) == v for k, v in LITERALS.items())
print(f"  all six deleted literals still reproduce : {ok}")
for k, v in LITERALS.items():
    print(f"    {k} -> {fused(*k, True)}  (literal {v})")
print(f"  opendde (12,48) -> {fused(12,48,True)}   (12,49) -> {fused(12,49,True)}   "
      f"<- the intended win, kept")
print(f"  rf3     (2,24)  -> {fused(2,24,True)}   <- was {fused(2,24,False)}, now refused")
print(f"  boltzgen(2,4)   -> {fused(2,4,True)}   <- was {fused(2,4,False)}, now refused")

import sys, torch
tt = torch.load(sys.argv[1] + "/inputs.pt", weights_only=False)["s_trunk"].float()
u = torch.load(sys.argv[2], weights_only=False)["s_trunk"].squeeze(0).float()
cm = u.abs().amax(0)
for lo, hi in [(0, 64), (64, 512), (512, 4096), (4096, 1e9)]:
    ch = (cm >= lo) & (cm < hi)
    a, b = tt[:, ch], u[:, ch]
    da, db = a - a.mean(0), b - b.mean(0)
    print(f"ch |up| in [{lo},{hi}): n={int(ch.sum())} rel err {float((a-b).norm()/b.norm()):.4g} "
          f"token-part rel err {float((da-db).norm()/db.norm()):.4g} up token std {float(db.std()):.4g} tt token std {float(da.std()):.4g}")

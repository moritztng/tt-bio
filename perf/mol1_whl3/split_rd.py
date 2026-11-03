import sys, torch
r = torch.load(sys.argv[1], weights_only=False); up = torch.load(sys.argv[2], weights_only=False)
small = up[-1].abs().amax(0) < 512
def tok(x): return x - x.mean(-2, keepdim=True)
for k, v in r.items():
    for j, x in enumerate(v if isinstance(v, list) else [v]):
        x = x.double().reshape(-1, x.shape[-1])
        line = f"{k}[{j}] {tuple(x.shape)} absmax {x.abs().max():.4g} std {x.std():.4g}"
        if x.shape[-1] == 384:
            line += f" | small absmax {x[:,small].abs().max():.4g} tokstd small {tok(x[:,small]).std():.4g} big tokstd {tok(x[:,~small]).std():.4g}"
        print(line)

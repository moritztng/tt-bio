"""Per-fold ttnn.matmul call census for OpenFold3.

Patches ttnn.matmul, records caller file:line plus padded shapes / dtypes / buffer types /
which config keyword (if any) the call passed, then runs a tt-bio predict. The first fold is
discarded so the counts come from a warm fold.

  E="TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:perfwar-of3-matmul-sites PYTHONPATH=$PWD"
  env $E $P perf/of3_mm/mm_census.py --out perf/of3_mm/census_openfold3_298.json --folds 2 -- \
      predict examples/prot300.yaml --model openfold3 --accelerator tenstorrent

Everything after `--` is handed to the tt-bio click CLI verbatim.
"""
import argparse, collections, json, sys, traceback


def install(counter, state):
    import ttnn
    real = ttnn.matmul

    def shape_of(t):
        for attr in ("padded_shape", "shape_with_tile_padding"):
            v = getattr(t, attr, None)
            if v is not None:
                try:
                    return [int(x) for x in v]
                except TypeError:
                    pass
        return [int(x) for x in t.shape]

    def patched(a, b, *args, **kw):
        if state["on"]:
            fr = sys._getframe(1)
            site = f"{fr.f_code.co_filename.split('/')[-1]}:{fr.f_lineno}"
            try:
                key = (site, tuple(shape_of(a)), tuple(shape_of(b)),
                       str(a.dtype), str(b.dtype),
                       str(a.memory_config().buffer_type), str(b.memory_config().buffer_type),
                       "core_grid" if kw.get("core_grid") is not None else
                       ("program_config" if kw.get("program_config") is not None else "none"))
                counter[key] += 1
            except Exception:
                counter[(site, "introspect-failed", "", "", "", "", "", "")] += 1
        return real(a, b, *args, **kw)

    ttnn.matmul = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=2,
                    help="how many times to run the CLI; only the last is counted")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    counter = collections.Counter()
    state = {"on": False}
    install(counter, state)

    from tt_bio.main import cli
    for i in range(a.folds):
        counter.clear()
        state["on"] = (i == a.folds - 1)
        try:
            cli.main(args=list(rest), standalone_mode=False)
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc()
            break

    rows = []
    for k, n in sorted(counter.items(), key=lambda kv: -kv[1]):
        site, ash, bsh, ad, bd, ab, bb, hint = k
        row = dict(site=site, a=list(ash) if isinstance(ash, tuple) else ash,
                   b=list(bsh) if isinstance(bsh, tuple) else bsh,
                   a_dtype=ad, b_dtype=bd, a_buf=ab, b_buf=bb, hint=hint, calls=n)
        if isinstance(ash, tuple) and len(ash) >= 2:
            B = 1
            for d in ash[:-2]:
                B *= d
            row.update(B=B, Mt=ash[-2] // 32, Kt=ash[-1] // 32, Nt=bsh[-1] // 32)
        rows.append(row)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{len(rows)} (site, shape) classes, {sum(counter.values())} calls -> {a.out}")


if __name__ == "__main__":
    main()

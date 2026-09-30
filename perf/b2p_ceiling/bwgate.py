"""triatt_bw's L1 gate priced on both grids, card-free, at BC2's shape [N, 4, N, 32]."""
import json, sys
import tt_bio.triatt_bw as T
out = []
for grid, name in (((8, 9), "WH 8x9"), ((11, 10), "BH 11x10")):
    for n in (192, 224, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608):
        best = T.largest_fitting_q_chunk(n, 4, n, 32, grid)
        p1 = T.plan(n, 4, n, 32, grid, q_chunk_tiles=1)
        Nt = n // 32
        out.append({"grid": name, "n": n, "Qt": None if best is None else best["Qt"],
                    "cb_at_Qt1": p1["l1_bytes"], "bias_plus_dbias": Nt * Nt * (2048 + 4096),
                    "budget": T.L1_PER_CORE - T.PROGRAM_RESERVE})
for r in out:
    print("{grid:9s} n={n:4d} largest_fitting_Qt={Qt!s:5s} CB_at_Qt1={cb_at_Qt1:8d} "
          "bias+dbias={bias_plus_dbias:8d} budget={budget}".format(**r))
json.dump(out, open(sys.argv[1], "w"), indent=1)

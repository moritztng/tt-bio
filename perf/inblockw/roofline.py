READ, WRITE, COPY, COMPUTE, BAL = 390.8, 264.0, 394.3, 101.00, 258.4
NOC0_FRAC = 171.1 / 283.3  # W3, qb1 card 2: per-NOC write roofs
NOC0 = WRITE * NOC0_FRAC

rows = [
    ("trimul out-proj 102400x256@256x256", 102400, 256, 256, 0.6789, 0.4444),
    ("pair transition down 102400x512@512x128", 102400, 512, 128, 1.0284, 0.5327),
    ("pair transition up 102400x128@128x512", 102400, 128, 512, 0.7772, 0.6937),
    ("trimul in-proj 102400x128@128x256", 102400, 128, 256, 0.4382, 0.3761),
    ("pair proj c_z 102400x128@128x128", 102400, 128, 128, 0.2988, 0.2267),
    ("pair->bias heads 102400x128@128x32", 102400, 128, 32, 0.1729, 0.1396),
    ("single transition up 320x768@768x3072", 320, 768, 3072, 0.0826, 0.0539),
    ("single transition down 320x3072@3072x768", 320, 3072, 768, 0.0593, 0.0512),
]
print(f"roofs card 2: compute {COMPUTE} TFLOP/s, read {READ}, write {WRITE}, copy {COPY} GB/s, "
      f"balance {BAL} FLOP/byte; NOC-0 write est {NOC0:.1f} GB/s\n")
hdr = (f"{'op':40s} {'AI':>7s} {'side':>7s} {'MB':>7s} "
       f"{'BASE GB/s':>10s} {'%copy':>6s} {'TUNED GB/s':>11s} {'%copy':>6s} "
       f"{'%NOCbnd':>8s} {'TFLOP/s':>8s}")
print(hdr)
for name, M, K, N, base, tuned in rows:
    fl = 2 * M * K * N
    by = (M * K + K * N + M * N) * 2
    ai = fl / by
    wr = M * N * 2
    rd = (M * K + K * N) * 2
    # bound with reads and writes overlapped, write pinned to the NOC-0 estimate
    bound_ms = max(rd / (READ * 1e9), wr / (NOC0 * 1e9)) * 1e3
    print(f"{name:40s} {ai:7.1f} {'mem' if ai < BAL else 'cmp':>7s} {by/1e6:7.1f} "
          f"{by/1e6/base:10.1f} {by/1e6/base/COPY*100:5.1f}% {by/1e6/tuned:11.1f} "
          f"{by/1e6/tuned/COPY*100:5.1f}% {bound_ms/tuned*100:7.1f}% {fl/1e9/tuned:8.1f}")

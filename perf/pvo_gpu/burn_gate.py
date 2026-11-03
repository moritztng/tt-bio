"""Thermal/clock gate, run ON the box before any timed arm. 150 s of bf16 matmul at full load; the box passes
only if the last 90 s hold the SM clock >= 1200 MHz (median) with no thermal or HW-slowdown throttle reason.
A100-80 DE 54695728 passed the power-limit check (400 W uncapped) and then sat at 84 C, SW Thermal Slowdown,
210-270 MHz for every fold, so the power limit alone does not qualify a host. Exit 0 = pass, 3 = reject."""
import statistics, subprocess, sys, threading, time
import torch

THERMAL = 0x08 | 0x20 | 0x40 | 0x80  # HW slowdown, SW thermal, HW thermal, HW power brake
rows, stop = [], threading.Event()

def sample():
    q = "clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active"
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().split(", ")
        rows.append((time.time(), int(out[0]), int(out[1]), float(out[2]), int(out[3], 16)))
        time.sleep(1)

a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
t = threading.Thread(target=sample); t.start(); t0 = time.time()
while time.time() - t0 < 150:
    for _ in range(20): a = (a @ b).clamp_(-1, 1)
    torch.cuda.synchronize()
stop.set(); t.join()
tail = [r for r in rows if r[0] - t0 >= 60]
clk = statistics.median(r[1] for r in tail); hot = sum(1 for r in tail if r[4] & THERMAL)
print(f"BURN median_sm={clk} MHz max_temp={max(r[2] for r in tail)} C max_power={max(r[3] for r in tail):.0f} W "
      f"thermal_samples={hot}/{len(tail)} reasons={sorted({hex(r[4]) for r in tail})}")
sys.exit(0 if clk >= 1200 and hot == 0 else 3)

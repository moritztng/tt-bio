# The "host-spin wedge" on this box is a PCIe containment event, and it costs 1 ms to detect

Measured on qb2, 2026-09-17, board `...410D` (dev2 + dev3), while both states were live on the
same host so the check below has a real negative control rather than an argued one.

## What the wedge actually is

A chip whose upstream PCIe port has **Memory Space Enable cleared** still opens. `/dev/tenstorrent/N`
is present, `fuser` shows the holder, `benchlock` is satisfied, `ps` shows a busy process. But with
memory-space decoding off the port stops forwarding MMIO to the endpoint, so every register read
the driver issues returns nothing. The consequences are the entire wedge signature that this
campaign has been chasing with minute-scale instruments:

* the bring-up `ttnn.add` on one `[32,32]` bf16 tile never completes -- the host polls a completion
  that cannot arrive, at 100 % CPU, with `syscr +0` and `syscw +0`;
* `tt_aiclk`, `tt_heartbeat` and the rest of the telemetry attributes fail their READ with
  `ENODATA` while the sysfs files still exist, so an existence check passes a dead chip through;
* `dmesg` carries `Failed to set initial power state: -5` once per open attempt and, on a reset
  attempt, `Failed to send ARC message for A0 state` + `Telemetry not available`.

## The register

    port 0000:00:01.1  COMMAND 0x0407   dev0, board ...4103   healthy
    port 0000:00:01.3  COMMAND 0x0407   dev1, board ...4103   healthy, ran the release gate throughout
    port 0000:00:01.4  COMMAND 0x0405   dev2, board ...410D   quarantined
    port 0000:00:01.5  COMMAND 0x0405   dev3, board ...410D   quarantined

`0x0407` = I/O space + memory space + bus master. `0x0405` drops **bit 1, Memory Space Enable**.

**Both values still carry bit 2, Bus Master Enable.** A bus-master test therefore PASSES a
quarantined port, which is the mistake this file exists to stop: the first version of the check
here tested bit 2, reported `ok` on both dead chips, and was caught only because the control was
run on all four nodes instead of being reasoned about. The bit is 1, not 2.

## The kernel says so itself, and names the port

    [27653.8]  tenstorrent 0000:03:00.0: QB quarantine: port 0000:00:01.4 COMMAND 0407 -> 0405; reboot required

17:57:11Z for dev2. A reboot clears it; a reset does not, and `tt-smi -r 2` at 18:03:00Z logged ARC
failures on `03:00.0` and left the port at `0x0405`.

## What it replaces

`dispatch_probe.py` (60 s, opens the device, leaves an orphan if it hangs) and
`wedge_check.py` (needs MINUTES of forward-progress history) both still have their uses -- a chip
can host-spin for other reasons, and a healthy port does not prove a healthy chip. But when the
port is quarantined, one config-space read answers in about a millisecond, opens nothing, needs no
lease and no lock, and distinguishes "needs a reboot" from "needs a reset", which the other two
cannot do at all.

`dispatch_probe.py` now runs it first and returns 3 without opening the device. `run.sh` refuses
on it before taking `benchlock`.

## Chip scope of `tt-smi -r`, independently corroborated

At 18:03:00Z `tt-smi -r 2` logged on `0000:03:00.0` only; `0000:04:00.0` logged **nothing**, and
the release gate's leg on `0000:02:00.0` ran through it untouched. That matches
`c12-orchestrator`'s pass-33 retraction, from a second event: on this box, this tt-smi, today,
`-r <n>` resets the named chip alone.

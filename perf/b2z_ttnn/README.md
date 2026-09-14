# b2z-ttnn-upgrade — can a newer ttnn close Boltz-2's 2.34x deficit?

`ARCH: WH` (whglx, `j10glx02`, 32-chip Wormhole Galaxy, UMD chip 2). Nothing here has been taken on
Blackhole yet.

## What is current

| | version | date |
|---|---|---|
| tt-bio pin (`pyproject.toml:107`) | `ttnn==0.68.0` | wheel published 2026-04-13 |
| latest ttnn release | `0.78.0` | 2026-09-05 |
| tt-metal `main` | `v0.79.0-dev20260911` | 2026-09-11 |

Ten minor versions and five months. cp310 wheels exist for every release and whglx runs Python
3.10.12, so no source build is needed to answer the first question.

## The API surface does not block the bump

`api_census.py` resolves every `ttnn.<attr>` that `tt_bio/` calls against the installed module.
147 distinct symbols, including the whole hand-written-kernel surface (`generic_op`,
`ProgramDescriptor`, `KernelDescriptor`, `CBDescriptor`, `TensorAccessorArgs`,
`experimental.minimal_matmul`, `SDPAProgramConfig`, `split_work_to_cores`).

**Missing on 0.78.0: none.** The single reported miss, `ttnn.libs`, is a path string in a comment
and is equally "missing" on 0.68.0. Existence is not signature compatibility, but the usual way a
stack bump kills a port -- a renamed or deleted op -- does not happen here.

## What does block it: UMD refuses to open ANY chip on this box

**tt-bio's pin is the last ttnn that can open a chip on this box.**

| ttnn | `ttnn.open_device` on UMD chip 2 |
|---|---|
| 0.68.0 | **OK** -- folds 512 aa normally, 41.355 s warm, CIF `2e3b25a5520c8ff8` |
| 0.69.0 | hangs, no return in 150 s |
| 0.70.1 | hangs, no return in 150 s |
| 0.71.2 | throws: `ETH core heartbeat check failed ... e7-0 (NOC0), post code: 2040000` |
| 0.72.0 | throws, same core, same post code |
| 0.75.0 / 0.76.0 / 0.77.0 / 0.78.0 | throws: `Timed out waiting for ETH heartbeat ... Stuck at 0xabcde4a4` |

Everything from 0.69 up dies in `ttnn.open_device` before tt-bio runs a single op:

```
RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: 231146142722699596,
              ETH core e7-0 (NOC0) to advance. Stuck at 0xabcde4a4
Location: tt_metal/third_party/umd/device/topology/topology_discovery.cpp:720
  tt::umd::TopologyDiscovery::eth_heartbeat_running(...)
  tt::umd::TopologyDiscovery::discover_remote_devices()
  tt::umd::TopologyDiscovery::create_ethernet_map()
  tt::umd::Cluster::Cluster(...)
```

Three facts make this a hardware-state finding rather than a flaky open:

1. **Same chip, same core, same value, every time** -- ASIC `231146142722699596`, ETH core `e7-0`,
   stuck reading `0xabcde4a4`, across four ttnn versions and 20+ minutes. `0xabcd` is a valid
   `BASE_FW_HEARTBEAT_SIGNATURE`, so UMD is reading live firmware whose counter has stopped: that
   ERISC is halted, not absent and not mis-versioned.
2. **It is not the chip we asked for.** `TT_VISIBLE_DEVICES=2` does not help, because
   `Cluster::Cluster` runs topology discovery over the whole 32-chip ethernet mesh before any
   visibility filter applies. One halted ETH core takes the entire box out for the new stack.
3. **It fails 55 ms into discovery**, inside `create_ethernet_map` and before any
   `Discovering from ASIC ID` line, so the core sits on the first local chip UMD touches: UMD chip
   0, `/dev/tenstorrent/16`. Not the chip we asked for.
4. **0.68.0 is unaffected** and folds normally on the same chip, minutes apart, because it has no
   heartbeat gate. The check was added between 0.68 and 0.71.
5. UMD's own log says this box's firmware bundle **19.6.0 is newer than the latest fully tested
   19.5.0** for wormhole_b0, so this is not an under-old-firmware case.

`TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1` does not help: that skips cores whose link is retraining,
and this core's link is up -- its firmware is simply not ticking.

UMD `main` has made the failure configurable
(`TopologyDiscoveryOptions::eth_fw_heartbeat_failure`, `Action::THROW` vs warn-and-continue), but
nothing on the 0.72-0.78 wheels exposes it through an environment variable, and no cluster-descriptor
override is reachable from Python either. There is no userspace way around it on these wheels.

## Files

* `stack_fold.py` — one arm of one round: N timed 512 aa folds on whichever stack the interpreter
  carries, cold fold discarded, one JSON line per fold with its ttnn version and CIF digest.
* `drive_ab.py` — the interleaved old/new driver. Two ttnn versions cannot share an interpreter, so
  the pairing is the round (old, new, old, new) rather than the process, and `--aa` adds a second
  old-stack process per round for a cross-process A/A floor.
* `api_census.py` — which `ttnn.<attr>` calls in `tt_bio/` resolve on the installed stack.

## Two hypotheses, and the test that separates them

* **H1, dead ERISC.** Chip 0's ETH core e7-0 firmware is genuinely halted, and a Galaxy-isolated
  `tt-smi -r 0 --no_reinit` restarts it.
* **H2, co-tenant takeover.** A live tt-metal process on chip 0 has loaded that ETH core for
  dispatch or fabric, which stops the base-firmware heartbeat. Galaxy tt-metal does dispatch on ETH
  cores, and chip 0 was folding for another task throughout this pass, so H2 fits every observation
  as well as H1 does.

The test is one open probe on a new stack while chip 0 carries no tt-metal process. If it opens, the
bump is not blocked at all and only needs a quiet box.

Blackhole is exempt either way: the same function carries an explicit carve-out,
`if (tt_device->get_arch() != ARCH::BLACKHOLE && !eth_heartbeat_running(...))`, so a p300c is very
likely to open 0.78 today -- and Blackhole is the architecture the published cell is on.

## Update: the new stack does run, and here is what it took

The heartbeat gate is not the end of the road. Three things, in order, and each is a real finding:

**1. UMD's heartbeat gate, worked around in three bytes.** `umd_patch.py` rewrites the prologue of
`tt::umd::TopologyDiscovery::eth_heartbeat_running` to `return true`, which is the effect upstream's
`TopologyDiscoveryOptions::eth_fw_heartbeat_failure != Action::THROW` already has and the released
wheels do not expose. Returning *false* is not enough: there is a second throw site in
`create_ethernet_map` (0.78 line 314) that raises when the helper reports a bad core, and it named a
**second** stuck core, `e9-0` with post code `3030000`, after `e7-0` was silenced. So at least two
ETH cores on that chip have a frozen heartbeat, not one. This patch is a screen instrument, it lives
in one private venv next to a `.orig` copy, and any number taken through it says so.

**2. SFPI.** 0.78.0 pins **7.72.0** (`tt_metal/sfpi-version`, sha256 verified against the release
asset); whglx has 7.35.3 in `/opt/tenstorrent/sfpi`, which is what every other task on the box uses.
tt-metal looks for `<runtime root>/runtime/sfpi` **before** `/opt/tenstorrent/sfpi`, so the new
toolchain goes inside the venv and the system one is untouched. That is the way to run two stacks on
one shared box, and it is worth knowing generally.

**3. tt-bio's own fused kernels do not compile on 0.78.** `tt_bio/kernels/triatt_sdpa` and friends
are written against 0.68's LLK headers and hit dozens of errors in `compute_common.hpp`:
`VectorMode`, `RoundMode`, `InputClamping`, `PackMode` and `DataCopyType` became typed enums where
they were ints or bools, `exp_tile`/`exp_tile_init` changed arity, `ReluConfig`'s constructor went
private, and `mm_block_init_short`, `_sfpu_reciprocal_` and
`llk_math_eltwise_unary_sfpu_binop_with_scalar` are gone. **This, not the Python op surface, is the
cost of moving the pin** — and it falls entirely on the hand-written kernels, which is exactly the
surface this campaign is adding to.

So the A/B runs with every fused kernel off **on both arms** (`FUSED_KERNEL_FLAGS` in
`drive_ab.py`). That makes it an A/B of the stock op path, which is the right question anyway: it
asks whether tt-metal is the limit, without our kernels in between.

One transient to know about: a 0.78 process that dies during JIT build can leave sysmem mapped, and
the next open fails with `Sysmem mapped at unexpected NOC address` from
`silicon_sysmem_manager.cpp:417`. It clears by itself; it is not a wedged chip.

## The answer

**ttnn 0.78.0 is 0.9968x on the 512 aa fold and folds the 298 aa control 18.3 A wrong.**

| | 0.68.0 | 0.78.0 |
|---|---|---|
| 512 aa warm fold (paired, adjacent, fused kernels off both sides) | 49.647 s | 49.808 s |
| 298 aa control vs the committed reference | 0.2947 A, plDDT 0.9087 — **pass** | — |
| 298 aa control vs the 0.68.0 arm | — | **18.3187 A, plDDT 0.4054 — reject** |
| ops bit-identical out of 20 probed | | **16** |

The 0.68.0 arm reproducing the committed reference at 0.2947 A is the control on the rig: isolated
venv, isolated clone, every fused kernel off, still inside the 0.35 A bar. So the 18.3 A is the
stack, not the setup.

Sixteen of twenty ops are bit-identical, including `matmul` at both the library default and explicit
HiFi4. The four that differ — `softmax`, `layer_norm`, `rms_norm`, fused SDPA — all reduce across
the last axis and differ by at most 6e-2, which is accumulation order rather than a bug. A
perturbation that small cannot move a structure 18 A: for scale, taking the whole pair track from
bf16 to bfp8_b moved this same control 1.50 A. So the cause lives in a path these default-config
calls do not reach, most likely the explicit `compute_kernel_config` / program-config / sharded
paths tt-bio actually uses. Those four ops are shared-engine, so the exposure is every model, not
just Boltz-2.

Two caveats that belong next to any number above. The fold ratio is one paired round, n=2 per arm:
the cross-process A/A floor on this shared box is **0.2967x** (two old-stack processes, same chip,
ten minutes apart: 49.647 s and 167.325 s), so nothing below about 1.2x is resolvable on whglx while
the swarm runs, and the pair above is only trustworthy because the two arms ran back to back. And
the 0.78.0 arm exits SIGSEGV after its CIFs are written — teardown only, but one more thing that
would have to be fixed before the pin could move.

**So: do not move the pin, and do not fork.** The upside is zero and the cost is a full rewrite of
every hand-written kernel against a changed LLK surface. The Python op surface is stable across ten
minor versions, so tracking upstream stays cheap whenever we choose to; a fork would pay maintenance
forever to avoid a problem we do not have.

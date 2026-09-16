# CPU export controls

CLOCK: CPU only. 1350 MHz is the campaign context, not a hardware measurement here.

PREDICTED: The existing message exporter preserves the reducer metadata while omitting timing CSV expansion and Python timing postprocessing. Predicted model cycles saved: zero.

MEASURED: Three message exports match the archived metadata byte-for-byte. These are single CPU-process observations from wait4, not fold performance.

| Archive | Input bytes | Metadata bytes | Calls | Export peak RSS, KiB | Export elapsed, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| dm_control | 33050506 | 45251 | 293 | 204280 | 0.100864 |
| smoke1 | 18240120 | 18986 | 41 | 136608 | 0.080705 |
| smoke2 | 17834466 | 18986 | 41 | 134008 | 0.080885 |

The full archived metadata decodes to 259,634 distinct raw global-call records with no unparsed messages; validation peak RSS was 162,428 KiB. Ordered decoded SHA256: `1f80f913ede92a16c61f859d8ac320687026d56b8b747e9f7314c9c968be327c`.

All 13 retained window results match the published JSON exactly after JSON type normalization, including all 1,514 programs and their graph/source/shape joins. Replay peak RSS was 697,272 KiB. The replay reuses archived clock validity, without claiming new clock calibration.

The bounded packager round-trips the full 26,969,586-byte metadata and 99,348,740-byte clock inputs against their original SHA256. Peak RSS was 22,064 KiB. Window replay consumes the newly packaged metadata, not a separate original copy.

The actual exporter also passed four refusal controls: oversized input refused before launch, output stopped at 8,192 bytes with SIGXFSZ, 128 MiB address-space cap stopped the larger small-trace export, and elapsed timeout killed/reaped its own child. None published a metadata CSV. The native memory-limit failure was SIGSEGV, recorded as STOP with core dumps disabled, not accepted coverage. Synthetic controls separately exercise CPU-time exhaustion, malformed metadata, duplicate IDs, missing cache entries, failed exit, output reuse, original-CLI hook wiring and archive integrity. Fifteen exporter/packager tests and twelve existing census/smoke reducer tests pass.

The largest tested trace is 33,050,506 bytes. The 2,919,007,827-byte failed-run trace was not retrieved or tested. No size-to-memory extrapolation is accepted; the whole-file Worker can still hit the 8 GiB cap on it.

VERDICT: GO for the bounded host-export prerequisite only. No production code changes. Parent review remains required before another device census. Full-fold closure, missing windows, profiler perturbation and the 10-second goal remain unresolved.

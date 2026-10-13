The link is live on main: [README, Community projects](https://github.com/moritztng/tt-bio/blob/a3eaea964/README.md#community-projects). It names proga and you, says what it does, and says what we checked: tt-bio driven with proga's exact oracle command line on a Blackhole card against `boltz2` and `protenix-v2`, readers matching key for key.

The one thing still worth doing on your side is passing `--override` in `TTBioOracle` (your `BoltzOracle` already does), so every candidate gets its own fold rather than generation 1's. Details and both measured iptm values are in the comment above.

The keys proga reads (`iptm`, `pair_chains_iptm`, `chains_ptm`, `status`, `*_pae.npz`, the `_pae.json` sidecar) are documented and stay stable; if one breaks, that is a bug on our side and we'll treat it as one.

Closing this because what you asked for has landed. Thanks for building proga and for the patience after the first close. If you add an oracle or want the entry reworded, open an issue or a PR and we'll update it.

# Why `af2ig-trunk-device` fails its floor

The leg's floor was recorded before the template pair stack moved onto the card, and the leg has
scored the on-card stack against it ever since.

`de780ab7` (2026-08-21 16:32Z, "af2 port: the template pair stack runs on card") runs the
template's two `PairBlock`s as ttnn bf16 instead of torch bf16. They sit upstream of all 52 trunk
blocks, so the trunk starts from a differently-rounded pair and the structure module amplifies it
until `structure_module#3/traj` crosses `PCC_BAR` 0.998. The committed floor
`docs/implementation-parity-data/af2ig-trunk-device.json` was recorded at `afad85a0`, six hours
earlier, when the stack was still in host torch.

Probed on qb2 p300c, each report scored by a HEAD-pinned `device_floor.py` against a HEAD-pinned
copy of the committed floor, so the instrument does not move with the checkout under test.

    report                     commit     arm              taps  scalars  pcc_min              verdict
    RE_afd40d8d_parent.json    afd40d8d   de780ab7^1          9        3  0.9964227723349416   GAP
    RE_de780ab7.json           de780ab7   the move           13        1  0.9891207189666669   FAIL
    RD_b89b3a3e.json           b89b3a3e   0a2b8233^1         47        6  0.9759355664445303   FAIL
    RA_ac4b73b9.json           ac4b73b9   0a2b8233 + JSON    13        2  0.9960112623229      FAIL
    RC_head_templatehost.json  fb2bc462   --template-host     9        2  0.9969005517791679   FAIL (envelope only)

`afd40d8d` returns the committed floor's `pcc_min` to all sixteen digits and scores GAP. Its
immediate child is FAIL, and the four taps it adds
(`structure_module#3/{final_affines,final_atom14_positions,final_atom_positions,traj}`) are the
release's L4 FAIL.

The `b89b3a3e`/`ac4b73b9` pair is a separate, transient excursion inside the same window: a
duplicate `linear_o.bias` application from the `ce464f0a` merge, 47 taps, fixed by `0a2b8233`.
That fix moved the leg toward the floor. Nothing in the range needs reverting.

`RC_head_templatehost.json` is the control `de780ab7` shipped for exactly this and nobody ran
against the floor: on the release tag candidate, `--template-host` reproduces the floor's nine
failing taps name for name with a better `pcc_min` (0.99690 vs 0.99642) and a lower envelope max
(9.115 vs 12.297).

`probe.py` takes a worktree, a card and a name, runs the leg's own argv
(`tap_gate.py --params ... --stage complex --device`) and scores it. Reuses section 71's
`grid_control/armrun2.py`.

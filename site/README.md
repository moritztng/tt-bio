# The 512 aa performance page

`index.html` is a standalone page comparing tt-bio's five structure-prediction models on Blackhole
against H200 and B200 at 512 residues: predictions per hour per server, server price, one fold on one
card, and predictions per hour per dollar.

Preview it:

    python3 -m http.server -d site 8099    # then open http://localhost:8099/

A `file://` open fails: the page fetches its data file, and Chrome blocks that from a file URL unless
you pass `--allow-file-access-from-files`.

## Updating a number

Edit `data/perf-512aa.json` and nothing else. Every seconds-per-fold figure lives there with its
source doc, commit, host, noise floor and parity anchor. The page derives predictions per hour and
predictions per dollar in the browser, so no derived value is ever stored and no HTML changes when a
benchmark moves.

One cell per model per platform, holding the fastest measured arm. `availability` says whether that
arm is on main or still on a branch and `parity` says what it costs; the Methods block builds its
merge-state and parity paragraphs from those two fields, so they cannot drift from the numbers.

A cell with no measurement gets `"status": "pending"` or `"status": "blocked"` plus a reason, and
renders as such. Do not fill it with an estimate.

## Publishing

Live at https://moritztng.github.io/tt-bio/. `.github/workflows/pages.yml` runs on
`workflow_dispatch` only, so a data change goes live when someone runs that workflow by hand:

    gh workflow run "Publish the perf page" --repo moritztng/tt-bio

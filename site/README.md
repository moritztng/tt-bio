# The TT-Bio page

`index.html` is a standalone page in two sections. First, what TT-Bio runs: one table per group,
structure and binding affinity prediction, protein design, protein embeddings, with the `--model` id
in the first column and one short cell per capability. That section is static markup in the HTML, so
it renders even if the data fetch fails. Second, performance and cost: tt-bio's five
structure-prediction models on Blackhole against H200 and B200 at 512 residues, with throughput per
dollar of a Galaxy Blackhole against a DGX H200 and a DGX B200 on purchase price and again on total
cost of ownership, predictions per hour per server, prediction time on one AI Processor, and what
each server costs to buy and to power. That section is driven by `data/perf-512aa.json`.

Preview it:

    python3 -m http.server -d site 8099    # then open http://localhost:8099/

A `file://` open fails: the page fetches its data file, and Chrome blocks that from a file URL unless
you pass `--allow-file-access-from-files`.

## Updating a number

Edit `data/perf-512aa.json` and nothing else. Every seconds-per-prediction figure lives there with
its noise floor and parity anchor, and every price and power rating lives there with a dated source
link. The page derives predictions per hour, cost per hour and both throughput-per-dollar charts in
the browser, so no derived value is stored and no HTML changes when a benchmark moves.

One cell per model per platform, holding the fastest measured arm that passes parity against its own
reference. Every Tenstorrent cell states which parity it holds in `parity`, and Methods lists them:
bit-exact, byte-identical, or a measured structural delta inside a bar that was fixed before the
number existed. A cell with no measurement gets `"status": "blocked"` plus a reason and renders as
"does not run". Do not fill it with an estimate.

`cost_model` holds the amortisation window, the electricity rate and the platform both charts are
indexed to. The window is 4 years and the rate is the US industrial average, and they move the total
cost of ownership chart only. The purchase-price chart is predictions per hour divided by the price
itself, with no window: spreading the price over a window multiplies both sides of that ratio by the
same number, so it cancels. Only a price change moves it.

`amortisation_basis` is the one-line justification for the window and renders next to the formula in
Methods. Change the window and change that line with it.

## Publishing

Live at https://moritztng.github.io/tt-bio/. `.github/workflows/pages.yml` runs on
`workflow_dispatch` only, so a data change goes live when someone runs that workflow by hand:

    gh workflow run "Publish the perf page" --repo moritztng/tt-bio

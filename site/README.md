# The TT-Bio page

`index.html` is a standalone page in two sections. First, what TT-Bio runs: one table per group,
structure and binding affinity prediction, protein design, protein embeddings, with the `--model` id
in the first column and one short cell per capability. That section is static markup in the HTML, so
it renders even if the data fetch fails. Second, performance and cost: tt-bio's six
structure-prediction models on Blackhole against H200, B200 and A100 at 512 residues, with throughput
per dollar of a Galaxy Blackhole against a DGX H200, a DGX B200 and a DGX A100 on purchase price and
again on total cost of ownership, predictions per hour per server, prediction time on one AI Processor, and what
each server costs to buy and to power. That section is driven by `data/perf-512aa.json`.

Preview it:

    python3 -m http.server -d site 8099    # then open http://localhost:8099/

A `file://` open fails: the page fetches its data file, and Chrome blocks that from a file URL unless
you pass `--allow-file-access-from-files`.

## Updating a number

Edit `data/perf-512aa.json` and nothing else. Every seconds-per-prediction figure lives there with
its noise floor and parity anchor, and every price and power rating lives there with a dated entry in
`sources`. The page derives predictions per hour, cost per hour and both throughput-per-dollar charts
in the browser, so no derived value is stored and no HTML changes when a benchmark moves.

A `sources` entry normally carries a `url` and renders as a link. Leave the `url` out and the entry
renders as plain text, "stated" rather than "accessed": that is for a figure this page assumes rather
than one it read somewhere, and the only one is the DGX A100's $175,000. Do not hang an assumption
off a vendor document's link.

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

## Adding a platform

A new accelerator needs four edits and no more: the platform object and its server counterpart in
the JSON, and in `index.html` its colour in `SER`, its card-to-box entry in `SERVER_OF`, and its id
in `CELLS`. `CELLS` is the single ordering every chart, legend and table reads, and `SERVERS` is
derived from it through `SERVER_OF`, so nothing else hardcodes a platform list. Every id in `CELLS`
must have a `SERVER_OF` entry pointing at a priced, power-rated box: without one the page looks up an
undefined platform and renders blank.

Adding a platform must not move any number already on the page. The way to know is to diff what the
page renders, not what the JSON says: dump the DOM before and after with headless Chrome and compare.

`amortisation_basis` is the one-line justification for the window and renders next to the formula in
Methods. Change the window and change that line with it.

## Publishing

Live at https://moritztng.github.io/tt-bio/. `.github/workflows/pages.yml` runs on
`workflow_dispatch` only, so a data change goes live when someone runs that workflow by hand:

    gh workflow run "Publish the perf page" --repo moritztng/tt-bio

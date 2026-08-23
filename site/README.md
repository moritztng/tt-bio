# The TT-Bio page

Two pages, both driven by `data/perf-512aa.json`.

`benchmarks/index.html` opens with what TT-Bio runs: one table per group, structure and binding
affinity prediction, protein design, protein embeddings, with the `--model` id in the first column
and one short cell per capability. That section is static markup, so it renders even if the data
fetch fails, which also means it goes stale silently -- check it against `tt_bio/main.py`'s model
tuples when a model lands. Then performance and cost: seven structure-prediction models, two binder
design, one binding affinity and six protein embedding on Blackhole against H200, B200 and A100 at
512 residues, with throughput per dollar of a Galaxy Blackhole against a DGX H200, a DGX B200 and a
DGX A100 on purchase price and again on total cost of ownership, predictions per hour per server,
prediction time on one AI Processor, sequences per second for the embedding rows, and what each
server costs to buy and to power.

`index.html` is the landing page. Its bar chart reads the same JSON and derives the same three
metrics, so it has no numbers of its own.

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
number existed. A cell with no measurement gets a `status` other than `measured` plus a reason, and the two
available are not the same claim: `"blocked"` with a `reason` renders as "does not run" and means the
model cannot run there, while `"not measured"` with a `detail` renders as "not measured" and means
nobody has paid for the run. Do not fill either with an estimate.

`cost_model` holds the amortisation window, the electricity rate and the platform both charts are
indexed to. The window is 4 years and the rate is the US industrial average, and they move the total
cost of ownership chart only. The purchase-price chart is predictions per hour divided by the price
itself, with no window: spreading the price over a window multiplies both sides of that ratio by the
same number, so it cancels. Only a price change moves it.

## Adding a section

A group of rows that shares the per-accelerator axis and has a price for its server goes in
`CATEGORIES` in `benchmarks/index.html`, which is how `design` and `affinity` are drawn. That one
line puts the rows in both throughput-per-dollar charts, the per-server chart and all three derived
tables, so it is only correct when the cost index the page is built on covers that workload.

The `embed` section is deliberately outside `CATEGORIES` and is the model for a section that is not.
Its unit is sequences per second, its timed region differs from the folding rows', and the cost index
is built on the DGX H200 at 512 aa folds, so an embedding forward has no server price or power figure
it could honestly carry. It gets its own band, its own chart and its own table, and it enters no cost
or per-server surface. `render_check.js` asserts that absence per row rather than trusting anyone to
remember it, so adding `embed` to `CATEGORIES` fails the check.

Every new section needs its row count added to `EXPECT_ROWS` in `render_check.js`. That table is the
only hardcoded thing in the file, and it is there because every other check is derived from the data:
a row deleted from the data is invisible to all of them, so the page would render, quietly stop
drawing the row, and the check would still exit 0.

## Adding a platform

A new accelerator needs the platform object and its server counterpart in the JSON, then in
`benchmarks/index.html` its colour in `SER`, its card-to-box entry in `SERVER_OF`, and its id in
`CELLS`. `CELLS` is the single ordering every chart, legend and table on that page reads, and
`SERVERS` is derived from it through `SERVER_OF`, so nothing else there hardcodes a platform list.
Every id in `CELLS` must have a `SERVER_OF` entry pointing at a priced, power-rated box: without one
the page looks up an undefined platform and renders blank.

`index.html` carries its own `SERVER_OF` and `CELLS` for the landing chart, because the two pages
share no script. They must stay in sync: add the platform to both, or the landing chart silently
drops the column while the benchmark page draws it.

Adding a platform, a row or a section must not move any number already on the page. The way to know
is to diff what the page renders, not what the JSON says: dump the DOM before and after with headless
Chrome and compare, both pages. Run `node benchmarks/render_check.js` and
`python3 ../perf/perf-page-host-device-publish/check_numbers.py` from the repo root as well; between
them they catch a row that stopped drawing and a host/device half that stopped adding up.

`amortisation_basis` is the one-line justification for the window and renders next to the formula in
Methods. Change the window and change that line with it.

## Publishing

Live at https://moritztng.github.io/tt-bio/. `.github/workflows/pages.yml` runs on
`workflow_dispatch` only, so a data change goes live when someone runs that workflow by hand:

    gh workflow run "Publish the perf page" --repo moritztng/tt-bio

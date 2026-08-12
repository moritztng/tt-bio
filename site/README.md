# The TT-Bio page

`index.html` is a standalone page in two sections. First, what TT-Bio runs: every model id grouped
into structure and binding affinity prediction, protein design, and protein embeddings. That section
is static markup in the HTML, so it renders even if the data fetch fails. Second, performance and
cost: tt-bio's five structure-prediction models on Blackhole against H200 and B200 at 512 residues,
with throughput per dollar on purchase price and again on purchase price plus electricity,
predictions per hour per server, prediction time on one AI Processor, and what each server costs to
buy and to power. That section is driven by `data/perf-512aa.json`.

Preview it:

    python3 -m http.server -d site 8099    # then open http://localhost:8099/

A `file://` open fails: the page fetches its data file, and Chrome blocks that from a file URL unless
you pass `--allow-file-access-from-files`.

## Updating a number

Edit `data/perf-512aa.json` and nothing else. Every seconds-per-prediction figure lives there with
its noise floor and parity anchor, and every price and power rating lives there with a dated source
link. The page derives predictions per hour, cost per hour and both throughput-per-dollar charts in
the browser, so no derived value is stored and no HTML changes when a benchmark moves.

One cell per model per platform, holding the fastest measured arm that is bit-exact or byte-identical
against its own reference. A cell with no measurement gets `"status": "blocked"` plus a reason and
renders as "does not run". Do not fill it with an estimate.

`cost_model` holds the amortisation window, the electricity rate and the platform the chart is
indexed to. Change one of those and every per-dollar figure moves with it.

## Publishing

Live at https://moritztng.github.io/tt-bio/. `.github/workflows/pages.yml` runs on
`workflow_dispatch` only, so a data change goes live when someone runs that workflow by hand:

    gh workflow run "Publish the perf page" --repo moritztng/tt-bio

# AbAg-XM deep-N — interactive findings page

A single static page. No backend, no build step, no dependencies. It reads one file,
`data/insights.json`, which is written by the analysis pipeline:

```
python3 scripts/abag_xm_insights/build_insights.py -o site/data/insights.json
```

Re-run that after the panel changes and the page picks up every new number, including the
headline tiles — nothing on the page is typed in by hand.

Look at it locally:

```
python3 -m http.server 8899 --directory site
```

## Deploying

The page uses only relative paths, so it works from any host and any subpath. Two options:

- **GitHub Pages from a folder** — point Pages at this directory on whichever branch you
  publish. `.nojekyll` is already here so Pages serves the files untouched.
- **Any static host** — copy `index.html` and `data/` and serve them.

Fonts come from Google Fonts; the page falls back to system sans and mono without them.

Not published anywhere yet. Publication is a separate decision, as is publication of the
underlying dataset, which is not in this repository.

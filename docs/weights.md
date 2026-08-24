# Weights and caches

Every model downloads its own weights on first use. `tt-bio weights` shows what this host
has, what it would load, and what is still missing:

```
$ tt-bio weights
ARTIFACT            MODEL          SOURCE   STATUS       SIZE  PATH
boltz2-conf         boltz2         hf-file  present     2.13G  /home/you/.boltz/boltz2_conf.ckpt
mols                boltz2         hf-file  present     3.42G  /home/you/.boltz/mols
esmc-6b             esmc-6b        hf-repo  present    23.66G  /home/you/.cache/huggingface/hub/models--biohub--ESMC-6B/...
rf3                 rf3            url      missing         -  /home/you/.boltz/rf3/rf3_foundry_01_24_latest_remapped.ckpt
openfold3           openfold3      manual   present     2.13G  /home/you/.boltz/of3-p2-155k.pt
openbind            openbind       manual   present     2.13G  /home/you/.boltz/of3-ob-2025-06-30-174k.pt
nesso1-ccd          nesso1         hf-repo  present     0.38G  /home/you/.cache/huggingface/hub/models--recursionpharma--nesso/...
...
25/28 present, 64.8 GiB on disk, 6.8 GiB to fetch (tt-bio weights --download)
```

- `tt-bio weights --download` fetches everything that is missing or damaged.
- `tt-bio weights --download boltz2 rf3` fetches just those models' artifacts.
- `tt-bio weights --prune` reclaims superseded Hugging Face revisions and staging
  leftovers, printing the bytes and asking first. Naming models (`--prune saprot-1.3b`)
  also drops those models' artifacts.

The list comes from `tt_bio/weights.py`, which holds one row per artifact: source, repo or
URL, destination, licence and env override. Anything that needs "every artifact we ship"
reads that registry, so the CLI, the docs and what a fold actually fetches cannot drift
apart.

## Where the files go

Two directories, and one knob that moves both:

| | Default | Holds |
|---|---|---|
| tt-bio cache | `~/.boltz` | the flat checkpoints: Boltz-2, Protenix-v2, BoltzGen, RF3, RFD3, the CCD molecule library, OpenFold3 |
| Hugging Face hub cache | `~/.cache/huggingface/hub` | whole-repo models: ESMFold2, ESMC, SaProt, OpenDDE, Nesso-1 |

Set `TT_BIO_CACHE` and both move under it (the hub cache lands in `$TT_BIO_CACHE/hf`).
That is the one to use on a shared box, a rented instance or a cluster where `$HOME` is
small. If you already set `HF_HOME` or `HF_HUB_CACHE` yourself, tt-bio leaves it alone.

`BOLTZ_CACHE` still moves the tt-bio half only, and `predict --cache` still overrides it
per run. Neither touches the hub cache.

A full set of weights is about 86 GiB per host.

One artifact is deliberately outside the registry: the ESM-2 650M encoder (2.4 GiB) that
`tt-bio affinity --model nesso1` uses to embed its protein. `transformers` fetches it into
the hub cache on first use, so `TT_BIO_CACHE` still moves it, but it will show up under
"not claimed by any model" in `tt-bio weights`.

## Overriding one artifact

Every row takes an env var named after it: `TT_BIO_` plus the artifact key upper-cased with
dashes turned into underscores. `boltz2-conf` is `TT_BIO_BOLTZ2_CONF`, `saprot-1.3b` is
`TT_BIO_SAPROT_1_3B`. Point it at a file (or, for a whole-repo model, a snapshot directory)
and that is what loads, with no download. `tt-bio weights` marks a row that an override is
driving.

The four names that predate the registry keep working and still win when both are set:
`PROTENIX_CKPT`, `OF3_CKPT`, `RF3_CKPT`, `OPENDDE_CKPT`.

## Why `present` is not just "the file is there"

A download killed mid-flight leaves a truncated multi-GB file. Gating on "does the path
exist" then treats it as complete and reuses it on every later run, which surfaces much
later as:

```
PytorchStreamReader failed reading zip archive: failed finding central directory
```

So `tt-bio weights` verifies rather than checks for existence: checkpoints and the molecule
bundle are zip archives, so reading the central directory decides it in milliseconds, and
that is the same check the fetch path uses. A row that fails shows as `corrupt` and is
re-fetched on next use instead of being loaded.

Downloads never write to the final path. Each one stages next to its destination, verifies
against the source's byte count and archive structure, and only then renames into place, so
an interrupt cannot leave a file that looks complete. Direct HTTP downloads stage in a
`.<name>.part` that resumes across runs, so an interrupted 3 GB fetch continues rather than
restarting.

Archives that get unpacked (the CCD molecule library, the RFD3 weight split) are built under
a staging directory and renamed in, and a source archive that gets discarded after extraction
is only deleted once the extracted output verifies. An already-populated directory is adopted
as-is after its contents check out, so upgrading tt-bio never re-downloads or re-extracts
what a host already has.

## OpenFold3 and OpenBind are the exceptions

tt-bio does not download either OpenFold3 checkpoint. The project is Apache-2.0 and upstream
states the parameters are free for academic and commercial use, but the consortium publishes
no separate parameter licence, so fetching them on a user's behalf is not ours to do. tt-bio
verifies the file it is handed and says so if the copy is truncated.

| `--model` | file | put it at, or point | download |
|---|---|---|---|
| `openfold3` | `of3-p2-155k.pt` | `~/.boltz/of3-p2-155k.pt`, `OF3_CKPT` | from the consortium |
| `openbind` | `of3-ob-2025-06-30-174k.pt` | `~/.boltz/of3-ob-2025-06-30-174k.pt`, `TT_BIO_OPENBIND` | `https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-ob-2025-06-30-174k.pt` |

The two are different models, not two revisions of one: OpenBind is upstream tag `v0.5.0`,
whose diffusion transformer moved a LayerNorm, so the preview2 weights do not load on it and
its weights do not load on preview2. Keep both files if you want both `--model` choices.

## X-Cell has no weights to fetch

`tt-bio perturb --model xcell` has no row in the table above and never will until upstream
ships a checkpoint. Xaira has published the architecture but not the parameters, so
`tt-bio weights --download xcell` has nothing to do and says so. `--architecture-only` runs
the real network on random weights, which measures its speed and nothing about its biology.
See [xcell.md](xcell.md).

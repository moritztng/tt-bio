```text
████████╗████████╗        ██████╗  ██╗  ██████╗
╚══██╔══╝╚══██╔══╝        ██╔══██╗ ██║ ██╔═══██╗
   ██║      ██║    █████╗ ██████╔╝ ██║ ██║   ██║
   ██║      ██║    ╚════╝ ██╔══██╗ ██║ ██║   ██║
   ██║      ██║           ██████╔╝ ██║ ╚██████╔╝
   ╚═╝      ╚═╝           ╚═════╝  ╚═╝  ╚═════╝
```

> [!IMPORTANT]
> **TT-Boltz is now TT-Bio**

TT-Bio runs [Boltz-2](https://github.com/jwohlwend/boltz), [ESMFold2](https://github.com/Biohub/esm), [Protenix-v2](https://github.com/bytedance/Protenix), [OpenFold3](https://github.com/aqlaboratory/openfold-3), and [OpenDDE](#structure-prediction) structure prediction, [BoltzGen](#design) and [RFdiffusion3](#design) binder/protein design, and [ESMC protein embeddings](#protein-embeddings-esmc), and [SaProt structure-aware protein embeddings](#structure-aware-protein-embeddings-saprot) on Tenstorrent Blackhole and Wormhole, supporting single-card and multi-card configurations (e.g. QuietBox with 4 cards or Galaxy server with 32 cards). Multiple machines can also be combined into a single prediction run.

**Benchmarks: [tt-bio.com](https://tt-bio.com)** has throughput and cost for every model against
NVIDIA DGX H200, B200 and A100. The [full benchmark page](https://tt-bio.com/benchmarks/) has the
measured seconds behind every figure, the fixtures they were run on, the run conditions and the cost
model.

## Accuracy

Every model TT-Bio serves is validated against its official reference implementation on the same input and reproduces it within that reference's own run-to-run noise. See [`docs/implementation-parity.md`](docs/implementation-parity.md) for the methodology, per-target results, and reproduction commands.

## Performance and cost

Predictions and designs per hour per server, and throughput per dollar of purchase price and of total cost of ownership, measured against NVIDIA DGX systems. See [the benchmark page](https://tt-bio.com/benchmarks/) for the numbers, the fixtures they were measured on, and the cost model behind them.

## Installation

Create a Python virtual environment with Python 3.10 or 3.12, install with the Tenstorrent extra, then install the matching Tenstorrent system dependencies.

```bash
python3.10 -m venv env
source env/bin/activate
pip install 'tt-bio[tenstorrent]'
tt-bio install-deps
```

`tt-bio install-deps` installs the Tenstorrent system dependencies that match this release. It may ask for your sudo password.

On a host without a Tenstorrent card, plain `pip install tt-bio` is enough: the Boltz-2 CPU/GPU path (`--accelerator cpu` / `--accelerator gpu`) and the CLI work without the Tenstorrent SDK. The other models run on Tenstorrent only.

### From GitHub / source
Pin to a tagged release, track nightly `main` (may be untested), or work from an editable clone:
```bash
pip install "tt-bio[tenstorrent] @ git+https://github.com/moritztng/tt-bio.git@v0.6.5"   # pinned release, see Releases for the latest
pip install "tt-bio[tenstorrent] @ git+https://github.com/moritztng/tt-bio.git@main"     # nightly
# or
git clone https://github.com/moritztng/tt-bio.git
cd tt-bio
pip install -e '.[tenstorrent]'
tt-bio install-deps
```
Drop the `[tenstorrent]` extra on a host without a Tenstorrent card.

### Optional: Build TT-Metal / TT-NN from Source
If you need to build from source, follow the [Tenstorrent Installation Guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).

### Verify Installation
```bash
tt-bio --help
tt-bio predict --help
tt-bio msa --help
```

## Basic Usage

### Structure Prediction

```bash
tt-bio predict examples/prot.yaml --model boltz2 --override
```

Every command names its model with `--model`:

- **`boltz2`**: folds complexes of proteins, DNA, RNA, and ligands and predicts binding affinity. MSA-dependent (uses an MSA by default).
- **`esmfold2`** / **`esmfold2-fast`**: fold a single protein sequence on-device, no MSA required (`esmfold2-fast` is the lighter, faster checkpoint).
- **`protenix-v2`**: folds complexes of proteins, RNA, DNA, and ligands (an AlphaFold3-family model, the [Protenix](https://github.com/bytedance/Protenix) reproduction); MSA-dependent for proteins (uses an MSA by default), and also emits a PAE/PDE matrix with `--write_pae`.
- **`openfold3`**: folds proteins, RNA and DNA (an AlphaFold3-family model, the [OpenFold3](https://github.com/aqlaboratory/openfold-3) reproduction); MSA-dependent (uses an MSA by default), with optional per-chain templates. Polymer chains only, ligands, covalent bonds and cyclic chains are not supported yet (raise a clear error). Weights come from the OpenFold consortium; point `OF3_CKPT` at them.
- **`saprot`**: structure-aware protein embeddings, an ESM-2 encoder over a fused amino-acid + Foldseek-3Di vocabulary (446 tokens). Needs a structure for the 3Di structural tokens (`--structure`); runs sequence-only without it. Use for variant-effect / mutation-fitness scoring and function prediction.
- **`opendde`** / **`opendde-abag`**: antibody-antigen co-folding built on the Protenix-v2 stack plus a structural-token expander; `opendde-abag` selects the antibody-antigen checkpoint. Protein-only for now; proteins are MSA-dependent (uses an MSA by default, like Protenix-v2).
- **`rf3`**: folds complexes of proteins, RNA, DNA, and ligands (an AlphaFold3-family model, [RoseTTAFold3](https://github.com/RosettaCommons/foundry) from the Institute for Protein Design); MSA-dependent for proteins (uses an MSA by default). Also handles non-canonical residues, covalent modifications, and cyclic chains, and writes AlphaFold3-style `<name>_summary_confidences.json` (pTM, ipTM, chain-pair PAE/PDE, ranking score) next to each structure. Weights download from the IPD on first use.

```bash
tt-bio predict examples/prot.fasta --model esmfold2-fast --fast
tt-bio predict examples/prot.yaml --model protenix-v2   # MSA on by default; NA/ligand chains are single-sequence
tt-bio predict examples/prot.fasta --model openfold3    # MSA on by default; set OF3_CKPT to the weights file
tt-bio predict examples/9dsg_abag.yaml --model opendde-abag   # antibody-antigen co-fold, MSA on by default
tt-bio predict examples/prot.yaml --model rf3            # MSA on by default; weights fetch from the IPD
tt-bio predict examples/prot.yaml --model rf3 \
    --partial_t 150 --partial_structure start.cif       # refine start.cif instead of folding from scratch
tt-bio predict targets.yaml --model rf3 --early_stop_plddt 0.5   # skip the rollout on hopeless targets
```

| Feature | Boltz-2 | ESMFold2 | Protenix-v2 | OpenFold3 | OpenDDE | RF3 |
|---|---|---|---|---|---|---|
| Input | protein/DNA/RNA/ligand complex | single protein | protein/DNA/RNA/ligand complex | protein/RNA/DNA (polymer-only) | protein complex (antibody-antigen) | protein/DNA/RNA/ligand complex |
| MSA | MSA-dependent (on by default) | single-sequence | proteins MSA-dependent (on by default), NA/ligand single-sequence | proteins MSA-dependent (on by default) | proteins MSA-dependent (on by default) | proteins MSA-dependent (on by default) |
| Affinity / potentials / templates | yes | no | no | templates only | no | no |
| Pocket / contact constraints | yes | no | no | no | no | no |
| Covalent `bond` constraints | yes | no | yes | no | yes | from the input structure |
| PAE/PDE output (`--write_pae`) | no | no | yes | no | no | in `_summary_confidences.json` |

Targets up to at least 1095 residues fold on a single 12 GiB Wormhole card, on every structure
model including OpenDDE, whose structural-token expander makes it the strictest case. The pair
track switches to row-blocked execution at a size threshold smaller targets never reach, so
their speed and numerics are untouched. See [docs/large-targets.md](docs/large-targets.md).
Perf levers are gated at several sequence lengths, not just one; the release gate re-checks the
ladder against a recorded baseline. See [docs/size-generality.md](docs/size-generality.md).

All structure models support the sampling, output-format, and scheduling options.
MSA, affinity, constraint, and auxiliary-output options apply only where listed
below. Each model downloads its weights automatically on first use, except
OpenFold3: fetch the consortium checkpoint yourself and point `OF3_CKPT` at it,
or put it at `~/.boltz/of3-p2-155k.pt`. `tt-bio weights` lists every artifact with
its status, size and path; `--download` prefetches, `--prune` reclaims disk. Set
`TT_BIO_CACHE` to move all of it (both `~/.boltz` and the Hugging Face cache, about
65 GiB) somewhere with room. See [docs/weights.md](docs/weights.md).

Boltz-2, Protenix-v2, OpenFold3, OpenDDE, and RF3 are MSA-dependent and use an MSA **by default**, a local
ColabFold DB (`~/.boltz/msa_db`) if one is set up (see [Offline MSA](#offline-msa-optional)),
otherwise the online ColabFold server. Sending sequences to the online server (`api.colabfold.com`)
leaves your machine; a one-line notice is printed when that fallback is used. Pass
`--msa_db_path` for a private offline database, or `--single_sequence` to deliberately fold
without an MSA (lower accuracy; for batch-screening orphan sequences). OpenDDE multi-chain
predictions still request paired MSAs from `--msa_server_url`; use `--single_sequence` to
prevent all network MSA requests. ESMFold2 is single-sequence.

`--fast` makes some operations use a lower-precision numeric format that runs faster. Accuracy is typically very close.

OpenDDE-abag matches the upstream checkpoint on the standard 1AHW
antibody-antigen target. Both implementations perform poorly on 9DSG.

`predict` accepts either a single YAML/FASTA file or a directory containing many input files.

A live display shows the progress of each target. Prediction uses up to one card
per pending target, labelled in the display (`quietbox:tt0`, `quietbox:tt1`, ...).
Models load once per active card and stay resident:

```bash
tt-bio predict proteins/ --model boltz2 --out_dir results --fast
```

Pass `--devices 0,1,2,3` to pick or limit the available cards. A single target
remains a single-card fold; additional cards increase throughput only when
multiple targets are queued.

If you have additional machines with Tenstorrent cards, you can add them to a
single run; see [Optional: Multi-Machine Prediction](#optional-multi-machine-prediction).

### Protein Embeddings (ESMC)

Turn protein sequences into ESMC language-model embeddings on-device (no
folding, no MSA). `DATA` is a FASTA file, a directory of them, a YAML
`{id: sequence}` mapping, or a bare sequence string:

```bash
tt-bio embed proteins.fasta --model esmc-600m --out_dir embeddings
tt-bio embed "MQIFVKTLTGKTITLEV..." --model esmc-600m   # one-off sequence
```

`--model` selects the ESMC variant (`esmc-300m`, `esmc-600m`, `esmc-6b`). For
each sequence you get its **per-residue** embeddings (`[length, d_model]`
float32, one row per amino acid, row order == input order) and a **pooled**
whole-sequence vector (`[d_model]` float32, `--pool mean`/`max`/`cls`).
`--out_dir` (default `./embeddings`) gets:

- `<id>.npz` per sequence: `per_residue`, `pooled` (+ `logits` with `--logits`); `--format npz`, default
- `embeddings.parquet`: pooled vectors, one row per sequence; `--format parquet`
- `manifest.json`: model/pool/shapes/dtype and which file holds each sequence

Add `--logits` for the per-residue amino-acid predictions (300M/600M only),
and `--fast` for the lower-precision weight path. Weights download automatically on
first use.

Sequences batch automatically on 300M/600M (`--batch_size`, default 8): a
padded, length-bucketed device forward per batch, masked so results are
identical to running each sequence alone. Single-sequence calls
(`--batch_size 1`, e.g. serving one sequence at a time) replay through a
captured device trace once a length bucket repeats, up to ~1.5x faster per
call on QuietBox-class hosts, bit-identical, no flags needed.

To embed a large batch faster, shard it across several cards with
`--devices 0,1,2,3`: one worker per card, results reassembled in input order
and identical to a single-card run:

```bash
tt-bio embed proteins.fasta --model esmc-600m --devices 0,1,2,3
```

**Measured, not assumed:** fanout only pays off when there's enough work per shard to amortize each worker's model-load and device-init cost. On small batches it can be flat or worse than a single card. `esmc-6b` scales to 4 cards on suitably large batches. Reach for `--devices` on large batches, not small ad-hoc jobs; use `--controller` (below) for repeated/production embedding.

For repeated/production embedding, submit to a persistent pool instead: a worker
loads its model once and keeps it resident across every call, so the reload cost
above is paid once per worker, not once per invocation:

```bash
tt-bio controller --listen 8765          # starts + keeps a worker per local card
tt-bio embed proteins.fasta --model esmc-6b --controller http://localhost:8765
```

The same capability is available from Python:

```python
from tt_bio import esmc

emb = esmc.embed("MQIFVKTLTGKTITLEV...", model="esmc-600m")[0]
emb.per_residue   # [L, d_model] float32
emb.pooled        # [d_model] float32

# Shard a large set across cards (data-parallel, order preserved):
embs = esmc.embed(sequences, model="esmc-600m", devices=[0, 1, 2, 3])
```

### Structure-Aware Protein Embeddings (SaProt)

SaProt is a structure-aware protein language model, an ESM-2 encoder over a fused
amino-acid + Foldseek 3Di vocabulary (446 tokens). Where ESMC is sequence-only, SaProt
also encodes local structure, so its embeddings and MLM logits reflect both sequence
and shape. Use it for variant-effect / mutation-fitness scoring and function prediction
when you have a structure (predicted or experimental: fold it with `tt-bio predict`
first, then score it with SaProt).

```bash
tt-bio saprot proteins.fasta --model saprot-650m --structure structs/ --out_dir embeddings
tt-bio saprot proteins.fasta --model saprot-650m                # sequence-only (3Di = '#')
tt-bio saprot proteins.fasta --model saprot-650m --devices 0,1    # data-parallel across 2 cards
```

`--structure` is a PDB/cif file (single sequence) or a directory of `<id>.pdb`/`<id>.cif`
files, one per FASTA id. The 3Di structural tokens are computed on host with
[Foldseek](https://github.com/steineggerlab/foldseek) (`conda install -c bioconda foldseek`,
or set `FOLDSEEK_BIN`); it runs off-device. Omit `--structure` for sequence-only mode
(lower accuracy for 35M/650M; the 1.3B works sequence-only).

For each sequence you get **per-residue** structure-aware embeddings (`[length, d_model]`
float32) and a **pooled** vector, plus per-residue MLM logits (`[length, 446]` with
`--logits`) over the fused vocabulary, the log-likelihoods used for zero-shot mutation
scoring. Output layout matches `tt-bio embed` (`<id>.npz` / `embeddings.parquet` /
`manifest.json`).

`--model` selects the variant (`saprot-35m`, `saprot-650m`, `saprot-1.3b`). `--devices 0,1,2,3`
shards the input across cards data-parallel (one pinned subprocess each, results reassembled in
input order), bit-exact vs single-card with `--batch_size 1`. Parity vs the reference HuggingFace
checkpoint, the multi-card bit-exactness check, and warm throughput are in
[`docs/saprot-parity.md`](docs/saprot-parity.md).

Python:

```python
from tt_bio import saprot

emb = saprot.embed(("MQIFVKTLTGKTITLEV...", "dweweaepvrdidi..."), model="saprot-650m")[0]
emb.per_residue   # [L, d_model] float32, structure-aware
emb.logits        # [L, 446] float32 (with return_logits=True)
```

### Weights

Weights download on first use, so nothing here is required. `tt-bio weights` is for when
you want to see or move them:

```bash
tt-bio weights                       # every artifact: status, size, resolved path
tt-bio weights --download            # prefetch everything (e.g. before going offline)
tt-bio weights --download boltz2     # or just one model's set
tt-bio weights --prune               # reclaim superseded revisions and leftovers
```

A full set is about 65 GiB. It lands in `~/.boltz` and the Hugging Face cache; set
`TT_BIO_CACHE` to put both somewhere with more room. Each artifact also takes its own
override, so `TT_BIO_BOLTZ2_CONF=/mnt/weights/boltz2_conf.ckpt` loads that file instead of
downloading. Rows show as `corrupt` if a download was interrupted, and are re-fetched rather
than loaded. See [docs/weights.md](docs/weights.md).

### Offline MSA (Optional)

Use this if you have enough disk and RAM and want local MSA.
This avoids external MSA server calls and is faster for repeated runs.

```bash
tt-bio msa
tt-bio predict examples/prot.yaml --model boltz2 --override
```

`tt-bio msa` downloads UniRef30 to `~/.boltz/msa_db` (~100GB download, ~500GB on disk after indexing). `predict` auto-detects this path.

To add EnvDB and use it in prediction:
EnvDB can improve MSA coverage when UniRef30 hits are weak, at higher disk/RAM cost.

```bash
tt-bio msa --db all
tt-bio predict examples/prot.yaml --model boltz2 --use_envdb --override
```

**Key Options:**
- `--override`: Re-run from scratch, ignoring cached files
- `--use_msa_server`: Generate MSA via ColabFold API
- `--msa_db_path`: Use a local database at a custom path (e.g. `--msa_db_path /data/colabfold_db`)
- `--use_envdb`: Include EnvDB in offline MSA (`tt-bio msa --db all`)
- `--accelerator=tenstorrent`: Use Tenstorrent hardware (default, or use `cpu`/`gpu`)
- `--fast`: Makes some operations use a lower-precision numeric format that runs faster; accuracy is typically very close
- `--debug`: Show all raw output from the hardware and libraries instead of the progress display
- `--debug --log`: Same as `--debug`, but also print what each device is currently working on

### Shared MSA Server (Optional)

Host the database on one machine and let others fetch MSAs from it over HTTP, so each prediction machine need not keep its own ~500GB copy.

```bash
# On the machine with the database:
tt-bio msa-server --listen 0.0.0.0:8765

# On any other machine (no local database needed):
tt-bio predict examples/prot.yaml --model protenix-v2 --msa_endpoint http://HOST:8765
```

The server runs the same offline `colabfold_search` and serves unpaired `{hash}.a3m`, with a shared cache and a search-concurrency cap (`--max_concurrent`). Add `--token` to require `Authorization: Bearer <token>`. `--msa_endpoint` applies to `--model esmfold2`, `protenix-v2`, `openfold3`, `opendde`, and `rf3`.

### Binding Affinity Prediction (Boltz-2)

Predict binding affinity for protein-ligand complexes:

```bash
tt-bio predict examples/affinity.yaml --model boltz2 --use_msa_server --override --affinity_mw_correction
```

The `--affinity_mw_correction` flag applies molecular weight correction for more accurate predictions.

An affinity run folds the complex and then runs a second model that has its own
64-block trunk, so it costs more than a structure-only fold. All of it runs on the
card. FKBP12+SB3 at the default affinity protocol (200 sampling steps, 5 affinity
samples, single sequence) takes about 206 s per ligand on one Blackhole p150a,
measured as a whole `tt-bio predict` invocation with model load included.
`--sampling_steps_affinity` and `--diffusion_samples_affinity` are the two flags that
move that wall most.

The affinity trunk runs in fp32 because the predicted log10(IC50) is sensitive to
activation precision, and that is not configurable. Earlier releases ran it in fp32
on the host CPU instead, which is why affinity used to take minutes per ligand and
looked CPU-bound.

### Input Format

ESMFold2 accepts protein inputs only. Protenix-v2 accepts proteins, DNA, RNA,
ligands, and covalent `bond` constraints. OpenFold3 accepts proteins, DNA and RNA
plus per-chain templates, and rejects ligands and constraints with a named error.
OpenDDE accepts proteins and ligands
and honors covalent `bond` constraints between them. Boltz-2 additionally supports affinity, pocket/contact constraints,
potentials, and user-supplied templates.

Create a YAML file describing your complex:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B
```

**Entity Types:**
- **Polymers** (`protein`, `dna`, `rna`): provide `sequence`
- **Ligands** (`ligand`): provide `smiles` or `ccd` code

**Multiple Identical Chains:**
```yaml
- protein:
    id: [A, B]  # Two identical chains
    sequence: ...
```

## Understanding Results

### Output Structure

```text
<model>_results_prot/   # e.g. protenix_results_prot, boltz2_results_prot
├── structures/
│   ├── prot.cif                      # Best-ranked predicted structure
│   └── prot_model_1.cif              # Additional samples (if diffusion_samples > 1)
├── results.json                      # One entry per target with confidence/affinity metrics
├── power_profile.csv                 # (optional, --report-energy)
├── power_profile.png                 # (optional, --report-energy)
├── prot_pae.npz                      # (optional, --write_pae)
├── prot_pde.npz                      # (optional, --write_pde)
└── prot_embeddings.npz               # (optional, --write_embeddings)
```

MSA results are cached in `<out_dir>/msa/` (default `./msa/`), keyed by sequence hash. The same protein sequence is never searched twice, even across different input files or runs. The MSA search uses all available CPU threads and keeps the database index memory-mapped for maximum speed.

### Confidence Scores

Each target entry in `results.json` contains confidence metrics. The fields below are Boltz-2's; Protenix-v2 and OpenFold3 report the same `confidence_score` / `ptm` / `iptm` / `plddt` (and `all_runs` when `--diffusion_samples` > 1, ranked best-first), while an ESMFold2 entry instead carries `plddt` (mean, 0-1), `ptm` when available, and `n_residues` / `n_chains`.

```json
{
    "id": "prot",
    "status": "ok",
    "confidence_score": 0.84,
    "ptm": 0.84,
    "iptm": 0.82,
    "complex_plddt": 0.84,
    "chains_ptm": {
        "0": 0.85,
        "1": 0.83
    },
    "pair_chains_iptm": {
        "0": {"0": 0.85, "1": 0.72},
        "1": {"0": 0.82, "1": 0.83}
    }
}
```

- `confidence_score`: Overall confidence (0-1, higher is better), calculated as 0.8 × `complex_plddt` + 0.2 × `iptm`. Models are ranked by this score. OpenFold3 uses its own upstream ranking score instead (0.8 × `iptm` + 0.2 × `ptm` + 0.5 × disorder − 100 × clash), so its values are not comparable to the other models'
- `ptm`: Predicted TM-score for complex (0-1)
- `iptm`: Interface TM-score (0-1)
- `complex_plddt`: Average per-residue confidence (0-1)
- `chains_ptm`: Per-chain TM-scores (0-1)
- `pair_chains_iptm`: Per-chain-pair interface TM-scores (0-1)

### Affinity Predictions

For affinity targets, the same `results.json` entry also contains:

```json
{
    "affinity_pred_value": 2.47,
    "affinity_probability_binary": 0.41,
    "affinity_pred_value1": 2.55,
    "affinity_pred_value2": 2.19,
    "affinity_probability_binary1": 0.50,
    "affinity_probability_binary2": 0.42
}
```

- `affinity_probability_binary`: Probability of binding (0-1). Use for hit discovery (higher = more likely to bind)
- `affinity_pred_value`: Predicted binding affinity as log10(IC50) in μM. Use for ligand optimization (lower = stronger binding). Only compare between known active molecules
- `affinity_pred_value1`, `affinity_pred_value2`: Individual model predictions for binding affinity
- `affinity_probability_binary1`, `affinity_probability_binary2`: Individual model predictions for binding probability
- `runtime_s`: Wall-clock seconds for the whole target, structure and affinity together. Affinity
  targets also carry `structure_runtime_s` and `affinity_runtime_s`; the affinity leg is normally the
  larger of the two by several times, so read the split before pricing a screen

## Advanced Usage

### Input Format Details

#### Proteins with Custom MSA
```yaml
- protein:
    id: A
    sequence: MVTPEGNVSLVDES...
    msa: ./path/to/msa.a3m
```

#### Proteins with Modifications
```yaml
- protein:
    id: A
    sequence: MVTPEGNVSLVDES...
    modifications:
      - position: 5
        ccd: PTR  # Modified residue code
```

#### Ligands
```yaml
- ligand:
    id: B
    smiles: 'CC1=CC=CC=C1'  # SMILES string
    # OR
    ccd: ATP                # CCD code
```

#### Constraints

Pocket and contact constraints are **Boltz-2 only** (they need a trained constraint embedder). Covalent `bond` constraints work with **Boltz-2, Protenix-v2, and OpenDDE**. OpenFold3 does not support any `constraints:` block yet and rejects one with a named error rather than folding without it.

**Pocket Constraints** (binding site):
```yaml
constraints:
  - pocket:
      binder: B              # Ligand chain
      contacts: [[A, 10], [A, 11], [A, 12]]  # Binding site residues
      max_distance: 6.0      # Angstroms (4-20A, default 6A)
      force: false           # Use potential to enforce (default: false)
```

**Contact Constraints:**
```yaml
constraints:
  - contact:
      token1: [A, 10]
      token2: [A, 50]
      max_distance: 8.0
      force: false
```

**Bond Constraints** (covalent link, e.g. a covalent inhibitor, glycosylation, or disulfide; works with Boltz-2, Protenix-v2, and OpenDDE):
```yaml
constraints:
  - bond:
      atom1: [A, 10, SG]     # [chain, residue, atom]
      atom2: [B, 1, C12]     # ligand atom by name; polymer atoms by residue
```

> **OpenDDE + covalent bonds:** OpenDDE honors a `bond` constraint between a protein
> residue and a ligand atom (the covalent-inhibitor case) or between two protein
> residues (a disulfide or crosslink). Both ride the same `token_bonds` machinery as
> Protenix-v2 and are honored in the output (device-verified against upstream OpenDDE
> within the reference's own seed noise floor); see `examples/opendde_covalent_ligand.yaml`
> and `examples/opendde_covalent_bond.yaml`.

#### Templates

Use experimental structures as templates:

```yaml
templates:
  - cif: ./template.cif
    chain_id: A
    template_id: A
    force: true              # Enforce template alignment
    threshold: 2.0           # Max deviation in Angstroms
```

OpenFold3 takes templates per protein chain instead, as a precomputed alignment
`.npz` (the format the upstream benchmark cache ships). There is no template
search; the referenced structures are fetched from RCSB, and a missing one is a
hard error rather than a silently dropped template. See
`examples/7xi5_tmpl.yaml`.

```yaml
sequences:
  - protein:
      id: A
      sequence: MSSATPDPAEILT...
      templates: ./templates.npz
```

### Command-Line Options

Model-specific options are labelled below.

**Common Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `boltz2` | `boltz2`, `esmfold2`, `esmfold2-fast` (single-sequence ESMFold2), `protenix-v2` (AlphaFold3-family folder; protein / RNA / DNA / ligand complexes), `openfold3` (AlphaFold3-family folder; protein / RNA / DNA polymers, optional templates, `OF3_CKPT` weights), `opendde` / `opendde-abag` (antibody-antigen co-folding on the Protenix-v2 stack plus a structural-token expander; `opendde-abag` selects the antibody-antigen checkpoint; protein-only for now), or `rf3` (RoseTTAFold3, AlphaFold3-family folder; protein / RNA / DNA / ligand complexes, non-canonical residues, cyclic chains) |
| `--out_dir` | `./` | Output directory |
| `--cache` | `~/.boltz` | Weight cache directory. Whole-repo models (ESMFold2, ESMC, SaProt, OpenDDE) use the Hugging Face cache; `TT_BIO_CACHE` moves both, see [docs/weights.md](docs/weights.md) |
| `--accelerator` | `tenstorrent` | **(Boltz-2)** `tenstorrent`, `cpu`, or `gpu`; other models run on Tenstorrent |
| `--recycling_steps` | model-specific | 3 for Boltz-2 and OpenFold3 (OpenFold3 runs recycles+1 = 4 trunk cycles, its upstream default); 10 for Protenix-v2/OpenDDE/ESMFold2 (the ESMFold2 paper's benchmark setting) |
| `--sampling_steps` | model-specific | Requested diffusion sampling steps: 200 for Boltz-2/Protenix-v2/OpenFold3/OpenDDE; 100 for ESMFold2 (executes 68 after the sigma-schedule clip, the paper's protocol) |
| `--diffusion_samples` | `1` | Number of structure samples |
| `--partial_t` | `0` | rf3 only. Schedule index the diffusion rollout starts at, so it refines `--partial_structure` instead of folding from scratch. Higher stays closer to that structure |
| `--partial_structure` | — | rf3 only. The `.cif`/`.pdb`/`.json` structure `--partial_t` refines. It supplies the sequences too, so no MSA is attached |
| `--early_stop_plddt` | — | rf3 only. Abandon a target after the first trunk recycle if its mean pLDDT is below this. Writes no structure; the results entry carries `early_stopped` |
| `--max_parallel_samples` | `5` | Diffusion samples denoised in one batched forward. Higher is faster but costs device memory linearly; lower it if a large target runs out of memory |
| `--output_format` | `cif` | `cif` or `pdb` |
| `--seed` | `0` | Random seed for the diffusion sampler |
| `--trace` | `False` | **(Protenix-v2/OpenDDE)** Replay a captured trace of the per-step diffusion device stream. Lossless, and removes the per-step host dispatch; reserves 1 GiB of device memory |
| `--diffusion_trace` | `False` | **(Boltz-2)** The same for Boltz-2's diffusion DiT stream |
| `--write_pde` | `False` | **(Protenix-v2/OpenDDE)** Write the PDE matrix per target |
| `--write_embeddings` | `False` | **(Protenix-v2/OpenDDE)** Write the `s`/`z` embeddings per target |
| `--override` | `False` | Re-run from scratch |
| `--use_msa_server` | auto | Use the online ColabFold API; auto-enabled for Boltz-2/Protenix-v2/OpenFold3/OpenDDE when no local DB is found |
| `--single_sequence` | `False` | **(Boltz-2/Protenix-v2/OpenFold3/OpenDDE)** Skip all MSA requests; lower accuracy |
| `--msa_endpoint` | — | Fetch unpaired MSAs from a `tt-bio msa-server`; OpenDDE pairing still uses `--msa_server_url` |
| `--write_pae` | `False` | **(Protenix-v2/OpenDDE)** Write the token-token PAE/PDE matrices to `<name>_pae.npz` |
| `--use_potentials` | `False` | **(Boltz-2)** Apply physical constraints |
| `--affinity_mw_correction` | `False` | **(Boltz-2)** Apply MW correction to affinity |
| `--num_devices` | `0` | Number of TT devices (0=all available) |
| `--device_ids`, `--devices` | — | Comma-separated TT device IDs (e.g. `0,2`); `--devices` is the shorter alias (matches `tt-bio embed`) |
| `--host_threads` | all cores | Total CPU threads this process may use, split across its cards. Set it when you run several single-card predicts side by side on one host: each one otherwise sizes its thread pools to every core and they fight for the CPU. Use cores ÷ concurrent predicts |
| `--fast` | `False` | Makes some operations use a lower-precision numeric format that runs faster; accuracy is typically very close |
| `--listen` | — | Accept worker connections from other machines; see [Multi-Machine Prediction](#optional-multi-machine-prediction) |
| `--report-energy` | `False` | **(Boltz-2)** Enables optional energy profiling for one TT device (requires `tt-mgmt` add-on); writes `power_profile.csv` and `power_profile.png` |
| `--energy-metric` | `both` | **(Boltz-2)** Choose power channel(s): `tdp`, `input`, or `both` |
| `--energy-sample-hz` | `20.0` | **(Boltz-2)** Sampling rate in Hz for both `power_w` and `input_power_w` channels |

**Affinity-Specific Options (Boltz-2):**

| Option | Default | Description |
|--------|---------|-------------|
| `--sampling_steps_affinity` | `200` | Sampling steps for affinity |
| `--diffusion_samples_affinity` | `5` | Number of affinity samples |

**MSA Options** (Boltz-2, Protenix-v2, OpenFold3, and OpenDDE use an MSA by default; ESMFold2 only when requested):

| Option | Default | Description |
|--------|---------|-------------|
| `--msa_db_path` | auto-detect | Path to local ColabFold database (`~/.boltz/msa_db` if present) |
| `--msa_dir` | `<out_dir>/msa` | MSA cache directory. Point it at a shared persistent path to reuse `{seq_hash}.a3m` across runs |
| `--msa_cache_only` | `False` | Treat `--msa_dir` as the only MSA source: never search, and fail rather than quietly fold a chain single-sequence |
| `--use_envdb` | `False` | Also search environmental database |
| `--use_msa_server` | auto | Use ColabFold API for MSA (auto-enabled when no local DB is found) |
| `--single_sequence` | `False` | Fold without an MSA (Boltz-2/Protenix-v2/OpenFold3/OpenDDE) |
| `--msa_server_url` | `https://api.colabfold.com` | MSA server URL |
| `--msa_pairing_strategy` | `greedy` | `greedy` or `complete` |
| `--max_msa_seqs` | `8192` | Maximum MSA sequences |
| `--subsample_msa` | `False` | Subsample MSA |
| `--num_subsampled_msa` | `1024` | Number of subsampled sequences |

**MSA Database Setup Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `uniref30` | `uniref30` (~500GB), `envdb` (~800GB), or `all` |
| `--path` | `~/.boltz/msa_db` | Where to store the databases |
| `--install-tools` | `True` | Auto-install missing `mmseqs`/`colabfold_search` |

### MSA Server Authentication

For `--use_msa_server`:

**Basic Authentication:**
```bash
export BOLTZ_MSA_USERNAME=myuser
export BOLTZ_MSA_PASSWORD=mypassword
tt-bio predict ... --model boltz2 --use_msa_server
```

**API Key Authentication:**
```bash
export MSA_API_KEY_VALUE=your-api-key
tt-bio predict ... --model boltz2 --use_msa_server
```

## Optional: Multi-Machine Prediction

Combine the cards across any mix of Tenstorrent machines (a workstation, one
or more QuietBoxes, one or more Galaxy servers) into a single run.

On the machine driving the run:

```bash
tt-bio predict ./proteins --model boltz2 --listen 8765 --use_msa_server --fast
```

On every additional machine, replace `HOST` with the driving machine's
hostname or IP:

```bash
tt-bio worker --connect http://HOST:8765
```

## Optional: Energy Measurement (Boltz-2)

Use `--report-energy` to profile energy during prediction:

```bash
tt-bio predict examples/686.yaml --model boltz2 --override --device_ids 0 --report-energy --energy-metric both --energy-sample-hz 5
```

Behavior:
- Select metric channel(s) with `--energy-metric` (`tdp`, `input`, `both`)
- Uses one sampling rate (`--energy-sample-hz`, default 20 Hz)
- Supports only Tenstorrent runs with one selected device
- Records two power channels when available:
  - `power_w`: `tt-mgmt` UMD telemetry power (TDP channel)
  - `input_power_w`: `tt-mgmt` UMD telemetry input power
- Requires optional `tt-mgmt` installation:
  - `git clone --recursive https://github.com/aperezvicente-TT/tt-mgmt.git`
  - `pip install -e ./tt-mgmt`
- Prints energy summary metrics for selected channels
- Always writes:
  - `power_profile.csv`
  - `power_profile.png`

## Design

Design new binders and protein structures from a target or motif specification: one command, two models:

```bash
tt-bio design examples/binder.yaml --model boltzgen --num_designs 10
tt-bio design specs.json --model rfd3 --from_pdb --out_dir designs/
```

| Model | Designs | Input |
|-------|---------|-------|
| `boltzgen` (default) | protein / peptide / nanobody / antibody binders against a target | design YAML, same entity grammar as `predict` |
| `rfd3` | all-atom structures: binders, motif scaffolding, nucleic-acid binders | JSON spec with contig strings |

**[BoltzGen](https://github.com/HannesStark/boltzgen)** designs binders against a target structure. The pipeline runs design → inverse folding → folding → analysis → filtering and writes the top-ranked binders to `<out_dir>/final_ranked_designs/`. Input grammar, protocols, pipeline subsets, and options: [`docs/boltzgen-design.md`](docs/boltzgen-design.md). Designability (scRMSD) QA: [`docs/boltzgen-designability.md`](docs/boltzgen-designability.md).

**[RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967)** (RFD3) is an all-atom generative model that designs new protein structures and sequences from a specification, rather than folding an existing one. Design modes, the contig-string input grammar, and current limitations: [`docs/rfd3-design.md`](docs/rfd3-design.md).

Each model downloads its weights automatically on first use and fans out across every available card (`--devices 0,2` restricts). `tt-bio gen` still works as a deprecated alias for `tt-bio design --model boltzgen`.

## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{stark2025boltzgen,
  author = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and Xie, Yuxin and Hur, Eunsu and O'Donnell, Timothy John and Bushuiev, Anton and U{\c c}ar, Talip and Passaro, Saro and Mao, Weian and Reveiz, Mateo and Bushuiev, Roman and Pluskal, Tom{\'a}{\v s} and Sivic, Josef and Kreis, Karsten and Vahdat, Arash and Ray, Shamayeeta and Goldstein, Jonathan T. and Savinov, Andrew and Hambalek, Jacob A. and Gupta, Anshika and Taquiri-Diaz, Diego A. and Zhang, Yaotian and Hatstat, A. Katherine and Arada, Angelika and Kim, Nam Hyeong and Tackie-Yarboi, Ethel and Boselli, Dylan and Schnaider, Lee and Liu, Chang C. and Li, Gene-Wei and Hnisz, Denes and Sabatini, David M. and DeGrado, William F. and Wohlwend, Jeremy and Corso, Gabriele and Barzilay, Regina and Jaakkola, Tommi},
  title = {BoltzGen: Toward Universal Binder Design},
  year = {2025},
  doi = {10.1101/2025.11.20.689494},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}

@misc{candido2026language,
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and Rao, Roshan and Lin, Zeming and Verkuil, Robert and others},
  title = {Language Modeling Materializes a World Model of Protein Biology},
  year = {2026},
  url = {https://biohub.ai/papers/esm_protein.pdf},
  note = {Preprint; ESMC / ESMFold2}
}

@misc{protenix2025,
  author = {{ByteDance AML AI4Science Team}},
  title = {Protenix: An AlphaFold3 Reproduction for Biomolecular Structure Prediction},
  year = {2025},
  url = {https://github.com/bytedance/Protenix}
}

@misc{openfold3,
  author = {{OpenFold Consortium}},
  title = {OpenFold3: An Open-Source Reproduction of AlphaFold3},
  year = {2026},
  url = {https://github.com/aqlaboratory/openfold-3}
}

@article{butcher2025rfdiffusion3,
  author = {Butcher, Jasper and Krishna, Rohith and Mitra, Raktim and Brent, Rafael Isaac and Li, Yanjing and Corley, Nathaniel and Kim, Paul T and Funk, Jonathan and Mathis, Simon Valentin and Salike, Saman and Muraishi, Aiko and Eisenach, Helen and Thompson, Tuscan Rock and Chen, Jie and Politanska, Yuliya and Sehgal, Enisha and Coventry, Brian and Zhang, Odin and Qiang, Bo and Didi, Kieran and Kazman, Maxwell and DiMaio, Frank and Baker, David},
  title = {De novo Design of All-atom Biomolecular Interactions with RFdiffusion3},
  year = {2025},
  doi = {10.1101/2025.09.18.676967},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022}
}
```

## License

tt-bio is released under the MIT License (see [`LICENSE`](LICENSE)) and is built on the MIT-licensed Boltz-2 / Boltz-1 code. It bundles third-party code, each under its upstream license: the ESMFold2 host-side reference under `tt_bio/_vendor/` (the `esm` pipeline, MIT, © Chan Zuckerberg Biohub; and the HuggingFace ESMFold2 model definition, Apache-2.0), the OpenFold3 host-side data pipeline under `tt_bio/_vendor/openfold3/` (Apache-2.0, OpenFold Consortium), the BoltzGen binder-design source under `tt_bio/boltzgen/` (MIT, © Hannes Stärk), and RF3's host featurizer under `tt_bio/_vendor/rf3/`, `tt_bio/_vendor/foundry/` and `tt_bio/_vendor/atomworks/` (BSD-3-Clause, University of Washington / Institute for Protein Design). Protenix-v2, OpenFold3's on-device model, RFdiffusion3, and RF3's on-device model are independent ttnn reimplementations (no upstream compute code is vendored); Protenix-v2's weights download from ByteDance's Hugging Face mirror under Apache-2.0, RFdiffusion3's and RF3's checkpoints download directly from the Institute for Protein Design (BSD-3-Clause), and OpenFold3's `of3-p2-155k.pt` is the consortium's ungated public parameter release, which you fetch yourself (the project is Apache-2.0, stated by upstream as free for academic and commercial use; the consortium publishes no separate parameter license). See [`NOTICE`](NOTICE) for sources, versions, and modifications.

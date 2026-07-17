```
████████╗████████╗        ██████╗  ██╗  ██████╗
╚══██╔══╝╚══██╔══╝        ██╔══██╗ ██║ ██╔═══██╗
   ██║      ██║    █████╗ ██████╔╝ ██║ ██║   ██║
   ██║      ██║    ╚════╝ ██╔══██╗ ██║ ██║   ██║
   ██║      ██║           ██████╔╝ ██║ ╚██████╔╝
   ╚═╝      ╚═╝           ╚═════╝  ╚═╝  ╚═════╝
```

> [!IMPORTANT]
> **TT-Boltz is now TT-Bio**

TT-Bio runs [Boltz-2](https://github.com/jwohlwend/boltz), [ESMFold2](https://github.com/Biohub/esm), [Protenix-v2](https://github.com/bytedance/Protenix), and [OpenDDE](#structure-prediction) structure prediction, [BoltzGen](#boltzgen) binder design, and [ESMC protein embeddings](#protein-embeddings-esmc) on Tenstorrent Blackhole and Wormhole, supporting single-card and multi-card configurations (e.g. QuietBox with 4 cards or Galaxy server with 32 cards). Multiple machines can also be combined into a single prediction run.

## Installation

Create a Python virtual environment with Python 3.10 or 3.12, install, then install the matching Tenstorrent system dependencies.

```bash
python3.10 -m venv env
source env/bin/activate
pip install tt-bio
tt-bio install-deps
```

`tt-bio install-deps` installs the Tenstorrent system dependencies that match this release. It may ask for your sudo password.

### From GitHub / source
Pin to a tagged release, track nightly `main` (may be untested), or work from an editable clone:
```bash
pip install "tt-bio @ git+https://github.com/moritztng/tt-bio.git@v0.3.0"   # pinned release, see Releases for the latest
pip install "tt-bio @ git+https://github.com/moritztng/tt-bio.git@main"     # nightly
# or
git clone https://github.com/moritztng/tt-bio.git
cd tt-bio
pip install -e .
tt-bio install-deps
```

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
- **`opendde`** / **`opendde-abag`**: antibody-antigen co-folding built on the Protenix-v2 stack plus a structural-token expander; `opendde-abag` selects the antibody-antigen checkpoint. Protein-only for now; proteins are MSA-dependent (uses an MSA by default, like Protenix-v2).

```bash
tt-bio predict examples/prot.fasta --model esmfold2-fast --fast
tt-bio predict examples/prot.yaml --model protenix-v2   # MSA on by default; NA/ligand chains are single-sequence
tt-bio predict examples/9dsg_abag.yaml --model opendde-abag   # antibody-antigen co-fold, MSA on by default
```

| Feature | Boltz-2 | ESMFold2 | Protenix-v2 | OpenDDE |
|---|---|---|---|---|
| Input | protein/DNA/RNA/ligand complex | single protein | protein/DNA/RNA/ligand complex | protein complex (antibody-antigen) |
| MSA | MSA-dependent (on by default) | single-sequence | proteins MSA-dependent (on by default), NA/ligand single-sequence | proteins MSA-dependent (on by default) |
| Affinity / potentials / templates | yes | no | no | no |
| Pocket / contact constraints | yes | no | no | no |
| Covalent `bond` constraints | yes | no | yes | no |
| PAE/PDE output (`--write_pae`/`--write_pde`) | no | no | yes | no |

Shared across every model: `--fast`, `--recycling_steps`, `--sampling_steps`, `--diffusion_samples`, `--output_format`, the MSA flags, and the multi-card / multi-machine flags. Each model downloads its weights automatically on first use.

Boltz-2, Protenix-v2, and OpenDDE are MSA-dependent and use an MSA **by default**, a local
ColabFold DB (`~/.boltz/msa_db`) if one is set up (see [Offline MSA](#offline-msa-optional)),
otherwise the online ColabFold server. Sending sequences to the online server (`api.colabfold.com`)
leaves your machine; a one-line notice is printed when that fallback is used. Pass
`--msa_db_path` for a private offline database, or `--single_sequence` to deliberately fold
without an MSA (lower accuracy; for batch-screening orphan sequences). ESMFold2 is single-sequence.

`--fast` makes some operations use a lower-precision numeric format that runs faster. Accuracy is typically very close.

OpenDDE's antibody-antigen accuracy is currently weak (a known gap under investigation; see [`docs/opendde-port.md`](docs/opendde-port.md) for details).

`predict` accepts either a single YAML/FASTA file or a directory containing many input files.

A live display shows the progress of each protein. On a multi-card machine such
as a QuietBox or Galaxy server, every card is used in parallel and labelled in
the display (`quietbox:tt0`, `quietbox:tt1`, ...). Models load once per card
and stay resident, so jobs flow through without per-protein reloads:

```bash
tt-bio predict proteins/ --model boltz2 --out_dir results --fast
```

By default every detected card is used; pass `--devices 0,1,2,3` to pick or limit
which cards a run fans across (matching `tt-bio embed`). Each job is an independent
single-card fold pinned to its card, so results are identical to running that target
alone; sharding only changes which chip folds which target.

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
identical to running each sequence alone.

To embed a large batch faster, shard it across several cards with
`--devices 0,1,2,3`: one worker per card, results reassembled in input order
and identical to a single-card run:

```bash
tt-bio embed proteins.fasta --model esmc-600m --devices 0,1,2,3
```

**Measured, not assumed:** fanout only pays off when there's enough work per shard to amortize each worker's model-load and device-init cost. On small batches it can be flat or worse than a single card. `esmc-6b` scales to 4 cards on suitably large batches. See [`docs/esmc-multicard-scaling.md`](docs/esmc-multicard-scaling.md) for the numbers before reaching for `--devices` on a small job.

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

The server runs the same offline `colabfold_search` and serves unpaired `{hash}.a3m`, with a shared cache and a search-concurrency cap (`--max_concurrent`). Add `--token` to require `Authorization: Bearer <token>`. `--msa_endpoint` applies to `--model esmfold2`/`protenix-v2`.

### Binding Affinity Prediction (Boltz-2)

Predict binding affinity for protein-ligand complexes:

```bash
tt-bio predict examples/affinity.yaml --model boltz2 --use_msa_server --override --affinity_mw_correction
```

The `--affinity_mw_correction` flag applies molecular weight correction for more accurate predictions.

### Input Format

ESMFold2 takes a plain protein FASTA or a YAML with one or more `protein` chains. The richer inputs below (ligands, affinity, DNA/RNA, constraints, and templates) are Boltz-2 features.

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

```
boltz_results_prot/
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

Each target entry in `results.json` contains confidence metrics. The fields below are Boltz-2's; Protenix-v2 reports the same `confidence_score` / `ptm` / `iptm` / `plddt` (and `all_runs` when `--diffusion_samples` > 1, ranked best-first), while an ESMFold2 entry instead carries `plddt` (mean, 0-1), `ptm` when available, and `n_residues` / `n_chains`.

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

- `confidence_score`: Overall confidence (0-1, higher is better), calculated as 0.8 × `complex_plddt` + 0.2 × `iptm`. Models are ranked by this score
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

Pocket and contact constraints are **Boltz-2 only** (they need a trained constraint embedder). Covalent `bond` constraints work with **Boltz-2 and Protenix-v2**.

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

**Bond Constraints** (covalent link, e.g. a covalent inhibitor, glycosylation, or disulfide; works with Boltz-2 and Protenix-v2):
```yaml
constraints:
  - bond:
      atom1: [A, 10, SG]     # [chain, residue, atom]
      atom2: [B, 1, C12]     # ligand atom by name; polymer atoms by residue
```

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

### Command-Line Options

Options apply to every model unless tagged **(Boltz-2)**.

**Common Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `boltz2` | `boltz2`, `esmfold2`, `esmfold2-fast` (single-sequence ESMFold2), `protenix-v2` (AlphaFold3-family folder; protein / RNA / DNA / ligand complexes), or `opendde` / `opendde-abag` (antibody-antigen co-folding on the Protenix-v2 stack plus a structural-token expander; `opendde-abag` selects the antibody-antigen checkpoint; protein-only for now) |
| `--out_dir` | `./` | Output directory |
| `--cache` | `~/.boltz` | **(Boltz-2)** model cache directory; ESMFold2 uses the Hugging Face cache |
| `--accelerator` | `tenstorrent` | **(Boltz-2)** `tenstorrent`, `cpu`, or `gpu`; ESMFold2 always runs on Tenstorrent |
| `--recycling_steps` | `3` | Number of recycling iterations |
| `--sampling_steps` | `200` | Diffusion sampling steps |
| `--diffusion_samples` | `1` | Number of structure samples |
| `--output_format` | `cif` | `cif` or `pdb` |
| `--override` | `False` | Re-run from scratch |
| `--use_msa_server` | auto | Use online ColabFold API for MSAs. Auto-enabled for Boltz-2/Protenix-v2 when no local DB is found; ignored by ESMFold2 unless opted in |
| `--single_sequence` | `False` | **(Boltz-2/Protenix-v2)** Fold without an MSA (skips local DB and online server); lower accuracy |
| `--msa_endpoint` | — | **(ESMFold2/Protenix-v2)** Fetch MSAs from a `tt-bio msa-server` at this URL instead of searching locally |
| `--use_potentials` | `False` | **(Boltz-2)** Apply physical constraints |
| `--affinity_mw_correction` | `False` | **(Boltz-2)** Apply MW correction to affinity |
| `--num_devices` | `0` | Number of TT devices (0=all available) |
| `--device_ids` | — | Comma-separated TT device IDs (e.g. `0,2`) |
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

**MSA Options** (Boltz-2 / Protenix-v2 use an MSA by default; ESMFold2 only when you opt in):

| Option | Default | Description |
|--------|---------|-------------|
| `--msa_db_path` | auto-detect | Path to local ColabFold database (`~/.boltz/msa_db` if present) |
| `--use_envdb` | `False` | Also search environmental database |
| `--use_msa_server` | auto | Use ColabFold API for MSA (auto-enabled when no local DB is found) |
| `--single_sequence` | `False` | Fold without an MSA (Boltz-2/Protenix-v2) |
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

## BoltzGen

[BoltzGen](https://github.com/HannesStark/boltzgen) designs protein binders against a target. The pipeline runs design → inverse folding → folding → analysis → filtering and writes the top-ranked binders to `<output>/final_ranked_designs/`.

```bash
tt-bio gen run examples/binder.yaml --num_designs 10
```

This automatically uses every available card (splitting the designs across them and merging the results) and writes to `./binder/`. Add `--device_ids 0,2` to run on specific cards only.

### Input Format

```yaml
entities:
  - protein:
      id: B
      sequence: 80..120         # designed chain, sampled length per design
  - file:
      path: target.cif          # target structure (path relative to this yaml)
      include:
        - chain:
            id: A
```

`80..120` randomises the binder length per design; a fixed integer pins it. Ligand, DNA, and RNA targets use the same YAML grammar as `tt-bio predict`. See the [BoltzGen examples](https://github.com/HannesStark/boltzgen/tree/main/example) for binding sites, scaffolds, and residue constraints.

### Protocols

`--protocol` sets defaults appropriate for the binder type.

| Protocol | Use for |
|----------|---------|
| `protein-anything` (default) | de-novo protein binder |
| `peptide-anything` | peptide binder |
| `nanobody-anything` | nanobody / VHH |
| `antibody-anything` | antibody |
| `protein-small_molecule` | binder against a small-molecule target (adds affinity step) |
| `protein-redesign` | re-design existing residues (e.g. symmetric dimers) |

### Running a Subset

`--steps` restricts the pipeline.

```bash
tt-bio gen run examples/binder.yaml --steps design --num_designs 10
tt-bio gen run examples/binder.yaml --output existing/ --steps analysis filtering
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--protocol` | `protein-anything` | Protocol; sets defaults appropriate for the binder type |
| `--num_designs` | `10000` | Number of binders to generate |
| `--budget` | `30` | Number of top designs kept after filtering |
| `--output` | `./<basename>/` | Output directory |
| `--steps` | (all) | Run only specific stages |
| `--config STEP key=val` | — | Override per-stage config (e.g. `--config design sampling_steps=200`) |
| `--device_ids` | all cards | Restrict to specific cards (e.g. `0,2`) |
| `--fast` | `False` | Use a lower-precision path for some ops (slightly lower precision, faster) |
| `--cache` | `~/.boltz/boltzgen` | Cache for downloaded weights |
| `--debug` | `False` | Disable live display; show raw stage output |
| `--debug --log` | `False` | Add per-stage progress markers |

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

tt-bio is released under the MIT License (see [`LICENSE`](LICENSE)) and is built on the MIT-licensed Boltz-2 / Boltz-1 code. It bundles third-party code, each under its upstream license: the ESMFold2 host-side reference under `tt_bio/_vendor/` (the `esm` pipeline, MIT, © Chan Zuckerberg Biohub; and the HuggingFace ESMFold2 model definition, Apache-2.0) and the BoltzGen binder-design source under `tt_bio/boltzgen/` (MIT, © Hannes Stärk). Protenix-v2 is an independent ttnn reimplementation (no upstream code is vendored) and its weights download from ByteDance's Hugging Face mirror under Apache-2.0. See [`NOTICE`](NOTICE) for sources, versions, and modifications.

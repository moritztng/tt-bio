# Talking points

For booth staff. Every number has a source in brackets, listed at the end. If someone asks for a
number that is not here, say you will find out; do not estimate.

## Start a conversation

- "Type your name on the keyboard. Every letter becomes an amino acid, and those chips fold it
  in about a second."
- "That cloud is the model's real working. Each step you see is a state the chip computed, not an
  animation."
- "Press Tab and you can watch the four chips: clock, power, temperature, live."

## Thirty seconds, for anyone

Proteins are the machines in every cell, and what a protein does depends on its 3D shape. Working
out that shape in a lab can take months. This AI predicts it from the sequence of letters alone,
and here it does it live on the four Tenstorrent chips under the table. The points are the model's
guesses, step by step, settling into the protein. The finished ribbon is coloured by how sure the
model is: blue is confident, orange is a guess. The same software is open source and runs as a service, so a lab can use it tomorrow.

## Two minutes, for an HPC engineer

The model on screen is **ESMFold2**, a protein-structure model that needs no database search, so
it runs fully offline. Each of the four Blackhole chips holds its own resident copy and folds one
protein at a time: four independent folds in flight, no fold split across chips. A 100-residue
protein takes about 2 s, 400 residues about 15 s, at the chip's top clock of 1350 MHz [1]. The
screen shows each fold's real time and the clock sampled during it, so the numbers are checkable.

What you see is the diffusion sampler's real trajectory: 15 states per ESMFold2 fold and about 200 per
Boltz-2 fold, streamed from the chip [2]. The screen holds each real state and counts them
("Sampler step 9 of 15"); nothing is invented between them. The last frame is the scored structure,
drawn as a cartoon coloured by pLDDT on the AlphaFold scale.

Each chip is a grid of 110 Tensix cores here (11x10), each with 1.5 MB of SRAM that software
manages directly, 165 MiB in total, and five small RISC-V cores per Tensix that move data and drive
the matrix and vector engines [3]. Data flows core to core over two on-chip networks instead of
through a cache hierarchy.

The honest comparison: one Blackhole chip is slower than one H200, 2.4x on Boltz-2 and 4.1x on
ESMFold2 at 512 residues [4]. The case for it is per dollar: a 32-chip Galaxy does 3.7x
(ESMFold2) to 9.3x (Boltz-2) the folds per purchase dollar of a DGX B200 [4].

## Ten questions this audience asks

**1. Is it really live?** Yes. The label says *Live on four Blackhole chips* when it is, and
*Recorded folds* when it is showing folds recorded earlier on this box. A visitor's name is always
folded live. Every time on screen comes with the clock measured during that fold [1].

**2. How does it compare to a GPU?** Per chip it is slower: at 512 residues Boltz-2 takes 17.3 s
against 7.3 s on an H200, ESMFold2 29.4 s against 7.3 s [4]. Per dollar of list price, a Galaxy
does 3.7x to 9.3x the folds of a DGX B200 [4]. We never claim faster than an H200.

**3. How much power?** Each chip draws about 30 W idle and 40 to 60 W while folding, with peaks to
144 W, read from the driver's power counters on this box [5]. The chip's power limit reads 125 W.
We have not measured the whole box at the wall. On rated power, a Galaxy (12 kW) does 0.9x to 2.2x
the folds per kW of a DGX B200 (14.3 kW) [4].

**4. What precision?** bfloat16 on the chip, with fp32 kept where a result is sensitive to it, such
as Boltz-2's affinity trunk [6]. Accuracy is checked against the original PyTorch model, not
bit for bit: for ESMFold2 the per-residue confidence matches with a correlation of 0.999, and the
structure differs from the reference by 2.15 Å, less than the reference differs from itself
between two random seeds (1.98 Å is its own spread) [7].

**5. Does it scale out?** Folding is many independent proteins, so it scales by adding chips: here
four folds run side by side. A Galaxy has 32 chips [4], and JapanFold serves from Galaxies.

**6. What are the chips doing right now?** Running the whole model on the device: the language
model reads the sequence, the trunk reasons about which residues touch, the diffusion sampler
places the atoms, and a confidence head scores the result. The chip rows name the phase as it
happens [2].

**7. Is it open source?** Yes. TT-Bio is MIT licensed at github.com/moritztng/tt-bio [6]. It runs
Boltz-2, ESMFold2, OpenFold3, RoseTTAFold3, BoltzGen and more.

**8. Can I run it?** On any Tenstorrent Blackhole card: `pip install 'tt-bio[tenstorrent]'` [6]. A
p150a card lists at $1,399 [4]. Without hardware, use JapanFold.

**9. What does it cost on JapanFold?** $0.26 per processor hour, charged only for chip time a job
holds, and $100 free to start without a card. One Boltz-2 fold of 512 amino acids cost $0.0062 on
their own runs [8]. The QR code on screen goes to japanfold.aiand.com.

**10. Why does my name fold?** 20 of the 26 letters are amino acids, and the screen shows which
stand-in it uses for the other six. The model folds any sequence;
a real protein folds confidently, a name mostly does not, and the screen says which by the model's
own confidence score.

## Sources

1. qb2, ESMFold2 warm folds on one chip, 2026-10-03: 50 aa 0.86 s, 100 aa 1.93 s, 200 aa 5.03 s,
   400 aa 14.77 s, AICLK median 1350 MHz during each fold (`engine/runs/live-chip0`,
   `state/sc26-engine.md`).
2. `PROTOCOL.md` and the engine's default-off trajectory hooks: 15 frames per ESMFold2 fold, 201
   per Boltz-2 fold including the starting noise.
3. qb2's p300c chips report an 11x10 Tensix grid; 1.5 MB L1 per core (`docs/part-l1-budgets.md`).
4. `site/data/perf-512aa.json`, the published benchmarks page, 512 residues, list prices and rated
   power as sourced there. Per-dollar and per-kW ratios computed from it as the page does.
5. qb2 hwmon `power1_input`, four chips, ten samples 2 s apart, 2026-10-03 01:43Z: 30 W at
   800 MHz idle, 33 to 144 W at 1312 to 1350 MHz folding; `power1_max` 125 W.
6. `README.md` of tt-bio: installation, model table, precision notes; `LICENSE` (MIT).
7. `docs/esmfold2-e2e-parity.md`: pLDDT PCC 0.9989, Kabsch RMSD 2.15 Å vs the reference's two-seed
   spread of 1.98 Å.
8. japanfold.aiand.com pricing section, read 2026-10-03.

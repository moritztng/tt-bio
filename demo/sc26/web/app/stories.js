// What the attract loop says about a protein: its name and one line. A fold that carries its own
// `name` or `story` on the stream (from the gallery's manifest) uses those instead; otherwise the
// sequence is looked up here. Sequences are engine/attract.json's.
const GALLERY = [
  { name: 'Trp-cage', story: 'One of the smallest proteins there is.',
    sequence: 'NLYIQWLKDGGPSSGRPPPS' },
  { name: 'Villin headpiece', story: 'A tiny protein that folds in a few millionths of a second.',
    sequence: 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF' },
  { name: 'Protein G B1 domain', story: 'A bacterial protein that grabs antibodies.',
    sequence: 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE' },
  { name: 'Ubiquitin', story: 'The tag a cell puts on proteins it wants destroyed.',
    sequence: 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG' },
  { name: 'Green fluorescent protein', story: 'The protein that makes a jellyfish glow.',
    sequence: 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK' },
];
const BY_SEQ = new Map(GALLERY.map(g => [g.sequence, g]));

export function describe(f) {
  const g = BY_SEQ.get(f.sequence);
  f.name ??= g?.name ?? null;
  f.story ??= g?.story ?? null;
  return f;
}

export const storyOf = (name) => GALLERY.find(g => g.name === name)?.story ?? '';

// A gallery protein folded instead of a name the booth will not show.
export const INSTEAD = GALLERY[3];

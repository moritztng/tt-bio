// A visitor's name as a protein sequence.
//
// The twenty common amino acids have one-letter codes covering 20 of the 26 letters. The other six
// are mapped here, in the open, to their closest common relative, so the model only ever sees the
// twenty and the screen always says what was substituted. Spaces and hyphens show as gaps and are
// dropped from the sequence; anything else is ignored. A name is repeated until the protein is at
// least MIN_RES long, because ten residues is nothing to look at.

export const AMINO = {
  A: 'Alanine', C: 'Cysteine', D: 'Aspartate', E: 'Glutamate', F: 'Phenylalanine', G: 'Glycine',
  H: 'Histidine', I: 'Isoleucine', K: 'Lysine', L: 'Leucine', M: 'Methionine', N: 'Asparagine',
  P: 'Proline', Q: 'Glutamine', R: 'Arginine', S: 'Serine', T: 'Threonine', V: 'Valine',
  W: 'Tryptophan', Y: 'Tyrosine',
};

// letter -> [folded as, what the letter really is]
export const SUBSTITUTE = {
  B: ['D', 'D or N'],
  Z: ['E', 'E or Q'],
  J: ['L', 'I or L'],
  U: ['C', 'Selenocysteine'],
  O: ['K', 'Pyrrolysine'],
  X: ['A', 'Unknown'],
};

export const MAX_CHARS = 24;
export const MIN_RES = 60;

// Accent-stripped upper case, so "Zoë" types as ZOE.
const fold = (s) => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toUpperCase();

export function parseName(text) {
  const cells = [];
  for (const ch of fold(text).slice(0, MAX_CHARS)) {
    if (AMINO[ch]) cells.push({ ch, aa: ch, label: AMINO[ch] });
    else if (SUBSTITUTE[ch]) cells.push({ ch, aa: SUBSTITUTE[ch][0], label: SUBSTITUTE[ch][1], sub: true });
    else if (ch === ' ' || ch === '-') cells.push({ ch: ' ', gap: true });
  }
  const unit = cells.filter(c => c.aa).map(c => c.aa).join('');
  const times = unit ? Math.ceil(MIN_RES / unit.length) : 0;
  const subs = [...new Map(cells.filter(c => c.sub).map(c => [c.ch, c.aa])).entries()];
  return { cells, unit, times, sequence: unit.repeat(times), subs };
}

// "O and U fold as K and C, their closest common relatives."
export function substitutionLine(subs) {
  if (!subs.length) return '';
  const list = (xs) => xs.length === 1 ? xs[0] : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];
  const from = list(subs.map(s => s[0])), to = list(subs.map(s => s[1]));
  return subs.length === 1 ? `${from} folds as ${to}, its closest common relative.`
    : `${from} fold as ${to}, their closest common relatives.`;
}

const WORDS = ['', 'once', 'twice', 'three times', 'four times', 'five times', 'six times', 'seven times',
  'eight times', 'nine times', 'ten times'];

export function lengthLine(p) {
  return `${p.sequence.length} amino acids: your name ${WORDS[p.times] ?? p.times + ' times'}.`;
}

// Mean pLDDT on 0..1 -> the one line under a visitor's result.
export function verdict(meanPlddt) {
  if (meanPlddt >= 0.7) return 'Your name folds, confidently.';
  if (meanPlddt >= 0.5) return 'Parts of your name fold.';
  return 'Most of your name stays a cloud. Real proteins fold because evolution kept them.';
}

// A booth guard, local and dumb on purpose. See blocklist.txt for the matching rule.
let BLOCK = null;
export async function loadBlocklist(url) {
  try {
    const t = await (await fetch(url)).text();
    BLOCK = t.split('\n').map(l => l.trim().toUpperCase()).filter(l => l && !l.startsWith('#'));
  } catch { BLOCK = []; }
}

export function blocked(text) {
  if (!BLOCK) return false;
  const up = fold(text), joined = up.replace(/[^A-Z]/g, ''), words = up.split(/[^A-Z]+/);
  return BLOCK.some(w => w[0] === '*' ? joined.includes(w.slice(1)) : words.includes(w));
}

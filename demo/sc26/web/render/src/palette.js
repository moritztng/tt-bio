// Colour that carries meaning.
//   chain:  position along the chain, start to end. The one idea a non-biologist can read:
//           a single string of beads, and you can watch its two ends find each other.
//   water:  which parts avoid water (warm) and which seek it (cool). Kyte-Doolittle scale.
//   confidence: the model's own per-residue confidence, when the stream carries it.

const lin = (hex) => [0, 2, 4].map(i => ((parseInt(hex.slice(i + 1, i + 3), 16) / 255) ** 2.2));

// indigo, blue, cyan, mint, rose: hues that stay clean when shaded (yellows go olive in shadow)
const CHAIN_STOPS = ['#4338ca', '#2563eb', '#06b6d4', '#5eead4', '#fb7185'].map(lin);

function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

const KD = { ILE: 4.5, VAL: 4.2, LEU: 3.8, PHE: 2.8, CYS: 2.5, MET: 1.9, ALA: 1.8, GLY: -0.4, THR: -0.7,
  SER: -0.8, TRP: -0.9, TYR: -1.3, PRO: -1.6, HIS: -3.2, GLU: -3.5, GLN: -3.5, ASP: -3.5, ASN: -3.5,
  LYS: -3.9, ARG: -4.5 };
const WATER_STOPS = ['#38bdf8', '#e2e8f0', '#ff6b5b'].map(lin);  // seeks water, neutral, avoids water

const CONF_STOPS = [[1.0, 0.45, 0.20], [1.0, 0.85, 0.35], [0.30, 0.75, 1.0], [0.10, 0.35, 1.0]];

export const VDW = { C: 1.7, N: 1.55, O: 1.52, S: 1.8, P: 1.8, SE: 1.9, H: 1.1 };

export function residueColors(topo, scheme) {
  const n = topo.nres, out = new Float32Array(n * 3);
  for (let r = 0; r < n; r++) {
    let c;
    if (scheme === 'water') c = ramp(WATER_STOPS, ((KD[topo.resName[r]] ?? 0) + 4.5) / 9);
    else if (scheme === 'confidence' && topo.confidence)
      c = ramp(CONF_STOPS, (topo.confidence[r] - 50) / 45);
    else c = ramp(CHAIN_STOPS, n > 1 ? r / (n - 1) : 0.5);
    out.set(c, r * 3);
  }
  return out;
}

// A partner that is not part of the sequence (haemoglobin's hemes) sits on residue indices past its
// end. It is drawn in ember; reading the residue table there gave NaN, which lit those atoms as
// black and then white discs.
const PARTNER = lin('#ffa866');

export function atomColors(topo, scheme) {
  const rc = residueColors(topo, scheme), out = new Float32Array(topo.natom * 3);
  for (let i = 0; i < topo.natom; i++) {
    const r = topo.atomResidue[i];
    out.set(r < topo.nres ? rc.subarray(r * 3, r * 3 + 3) : PARTNER, i * 3);
  }
  return out;
}

export function atomRadii(topo) {
  const out = new Float32Array(topo.natom);
  for (let i = 0; i < topo.natom; i++) out[i] = VDW[topo.element[i]] ?? 1.7;
  return out;
}

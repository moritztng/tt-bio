// Colour that carries meaning, in the conventions a structural biologist already reads.
//   plddt:    the model's per-residue confidence on the AlphaFold scale, its four bands and colours
//             exactly as the AlphaFold Protein Structure Database draws them.
//   spectrum: position along the chain, blue at the start to red at the end (PyMOL's spectrum).
// Points during the fold are one sober grey: confidence does not exist until the fold is done.
// Atoms outside the protein (ligands, hemes) take the usual element colours.

const lin = (hex) => [0, 2, 4].map(i => ((parseInt(hex.slice(i + 1, i + 3), 16) / 255) ** 2.2));

export const PLDDT_BANDS = [   // [lower bound, colour, label]
  [90, '#0053D6', 'Very high (pLDDT > 90)'],
  [70, '#65CBF3', 'Confident (90 > pLDDT > 70)'],
  [50, '#FFDB13', 'Low (70 > pLDDT > 50)'],
  [0, '#FF7D45', 'Very low (pLDDT < 50)'],
];
const BANDS = PLDDT_BANDS.map(([lo, hex]) => [lo, lin(hex)]);

export const POINT = lin('#C4C9D1');
export const POINT_SIDE = lin('#5C626C');
export const GROUND = '#08090C';   // the app's --ground, so the canvas has no edge

const SPECTRUM = ['#2c4bd6', '#22a6d6', '#3fbf6e', '#e8c63a', '#e0582f'].map(lin);
function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
  return [0, 1, 2].map(k => stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f);
}

const CPK = { C: '#909090', N: '#3050F8', O: '#FF0D0D', S: '#FFFF30', P: '#FF8000', FE: '#E06633', MG: '#8AFF00',
  ZN: '#7D80B0', CA: '#3DFF00', CL: '#1FF01F', F: '#90E050', BR: '#A62929', I: '#940094', SE: '#FFA100' };
export const elementColor = (el) => lin(CPK[el] ?? '#FF1493');

// pLDDT arrives as 0-1 (the stream) or 0-100; per residue index, linear RGB.
export function residueColors(topo, scheme) {
  const n = topo.nres, out = new Float32Array(n * 3), conf = topo.confidence;
  const k = conf && Math.max(...conf) <= 1.01 ? 100 : 1;
  for (let r = 0; r < n; r++) {
    const c = scheme === 'plddt' && conf ? BANDS.find(([lo]) => conf[r] * k > lo || lo === 0)[1]
      : ramp(SPECTRUM, n > 1 ? r / (n - 1) : 0.5);
    out.set(c, r * 3);
  }
  return out;
}

export const VDW = { C: 1.7, N: 1.55, O: 1.52, S: 1.8, P: 1.8, SE: 1.9, H: 1.1 };

export function atomRadii(topo) {
  const out = new Float32Array(topo.natom);
  for (let i = 0; i < topo.natom; i++) out[i] = VDW[topo.element[i]] ?? 1.7;
  return out;
}

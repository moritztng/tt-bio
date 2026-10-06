// The demo's secondary-structure string (H/E/C per residue) for a recording's final structure.
//   node demo/booth/science/ss.mjs <recording.jsonl>
import { readFileSync } from 'fs';
import { backbone, secondaryStructure } from '../web/render/src/cartoon.js';
const L = readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const st = L.find(m => m.type === 'fold_start'), done = L.find(m => m.type === 'fold_done');
const topo = { natom: st.atoms.name.length, atomName: st.atoms.name, atomResidue: Int32Array.from(st.atoms.residue) };
const x = new Float32Array(Buffer.from(done.xyz, 'base64').buffer.slice(0));
console.log(secondaryStructure(backbone(topo), x).ss.join(''));

// ?walk=<name>&after=<s> : a stranger's run for the recording. Waits, types the name at a
// person's pace through the same keydown path a keyboard uses, pauses, presses Enter. Synthetic
// key events, so say so wherever the recording is shown.
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const key = (k) => dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));

export async function run(name, after, enter = true) {
  await sleep(after * 1000);
  for (const ch of name) { key(ch); await sleep(220 + Math.random() * 260); }
  if (!enter) return;
  await sleep(1400);
  key('Enter');
}

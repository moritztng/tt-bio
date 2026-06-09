// Parse bulk sequence input (multi-record FASTA or CSV) into {name, sequence}
// records, client-side, so thousands of proteins can be uploaded at once.

export function parseFasta(text) {
  const records = [];
  let name = null;
  let seq = [];
  const flush = () => {
    if (name !== null) records.push({ name: name || `seq_${records.length + 1}`, sequence: seq.join("") });
  };
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith(">")) {
      flush();
      const header = line.slice(1).trim();
      name = (header.split(/[|\s,]/)[0] || "").trim();
      seq = [];
    } else if (line.trim()) {
      seq.push(line.trim().replace(/\s+/g, ""));
    }
  }
  flush();
  return records.filter((r) => r.sequence);
}

export function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  if (!lines.length) return [];
  const split = (l) => l.split(",").map((c) => c.trim().replace(/^"|"$/g, ""));
  const header = split(lines[0]).map((h) => h.toLowerCase());
  const hasHeader = header.some((h) => ["name", "id", "sequence", "seq"].includes(h));
  let nameIdx = 0;
  let seqIdx = 1;
  if (hasHeader) {
    const ni = header.findIndex((h) => h === "name" || h === "id");
    const si = header.findIndex((h) => h === "sequence" || h === "seq");
    if (ni !== -1) nameIdx = ni;
    if (si !== -1) seqIdx = si;
  }
  const rows = hasHeader ? lines.slice(1) : lines;
  return rows.map((l, i) => {
    const cells = split(l);
    return {
      name: (cells[nameIdx] || `seq_${i + 1}`).trim(),
      sequence: (cells[seqIdx] || "").replace(/\s+/g, ""),
    };
  }).filter((r) => r.sequence);
}

// Decide by content/extension, then parse.
export function parseSequences(text, filename = "") {
  const looksCsv = /\.csv$/i.test(filename) || (!text.includes(">") && text.includes(","));
  let recs = looksCsv ? parseCsv(text) : parseFasta(text);
  // Fallback: a bare sequence or a plain newline-separated list of sequences
  // (no FASTA headers, no CSV). Only when nothing else matched and the input
  // isn't FASTA/CSV, so we never mask a malformed-FASTA error.
  if (!recs.length && !text.includes(">") && !looksCsv) {
    recs = text.split(/\r?\n/)
      .map((l) => l.replace(/\s+/g, ""))
      .filter((l) => /^[A-Za-z]+$/.test(l))
      .map((seq, i) => ({ name: `seq_${i + 1}`, sequence: seq }));
  }
  return recs;
}

// Turn a record into a per-target input file for the chosen model.
export function recordToTarget(rec, model) {
  if (model && model.startsWith("esmfold")) {
    // Chain id is always "A" — the record name becomes the file name (and thus
    // the result id). Using a long arbitrary name as the chain id breaks tt-bio.
    return { name: rec.name, content: `>A|protein\n${rec.sequence}\n` };
  }
  return {
    name: rec.name,
    content: `version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: ${rec.sequence}\n`,
  };
}

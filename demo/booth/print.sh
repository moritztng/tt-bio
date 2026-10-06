#!/usr/bin/env bash
# Print BOOTH.md and TALKING.md to A4 PDFs beside them, in the app's typeface.
#
#   demo/booth/print.sh          needs pandoc and google-chrome (or chromium)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
chrome=$(command -v google-chrome || command -v chromium || command -v chromium-browser)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/style.css" <<CSS
@font-face { font-family: Inter; src: url("file://$here/web/app/fonts/InterVariable.woff2"); }
@page { size: A4; margin: 12mm 14mm; }
body { font: 9.4pt/1.38 Inter, sans-serif; color: #10141c; max-width: none; margin: 0; }
h1 { font-size: 17pt; font-weight: 650; margin: 0 0 3pt; letter-spacing: -0.01em; }
h2 { font-size: 10.5pt; font-weight: 650; margin: 9pt 0 3pt; color: #0b6f86; }
p, ol, ul { margin: 0 0 4pt; } li { margin: 0 0 1.5pt; } ol, ul { padding-left: 15pt; }
table { border-collapse: collapse; width: 100%; margin: 2pt 0 4pt; }
th, td { text-align: left; vertical-align: top; padding: 2.5pt 6pt 2.5pt 0; border-bottom: 0.5pt solid #ccd3dc; }
th { font-weight: 600; color: #4a5566; } td:first-child { width: 30%; padding-right: 10pt; } a { color: inherit; text-decoration: none; }
code { font: 8.4pt ui-monospace, monospace; }
.foot { margin-top: 8pt; font-size: 7.8pt; color: #4a5566; }
CSS
for doc in BOOTH TALKING; do
  pandoc "$here/$doc.md" -f gfm -t html5 -s --metadata title="$doc" --css "$tmp/style.css" \
    -V pagetitle="$doc" -o "$tmp/$doc.html"
  sed -i '/<header id="title-block-header">/,/<\/header>/d' "$tmp/$doc.html"
  "$chrome" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
    --print-to-pdf="$here/$doc.pdf" "file://$tmp/$doc.html" 2>/dev/null
  echo "$doc.pdf: $(pdfinfo "$here/$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2}') page(s)"
done

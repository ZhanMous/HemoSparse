#!/usr/bin/env bash
set -euo pipefail

# docker_convert.sh
# 在容器内将 HemoSparse_presentation.pptx 转为 PDF 并放到 outputs/ 下

PPT="HemoSparse_presentation.pptx"
OUTDIR="outputs"

cd "$(dirname "$0")" || exit 1

if [ ! -f "$PPT" ]; then
  echo "Error: $PPT not found in $(pwd)"
  exit 2
fi

mkdir -p "$OUTDIR"

echo "Converting $PPT -> PDF (headless LibreOffice)..."
libreoffice --headless --convert-to pdf --outdir "$OUTDIR" "$PPT"

OUTPDF="$OUTDIR/${PPT%.pptx}.pdf"
if [ -f "$OUTPDF" ]; then
  echo "Conversion successful: $OUTPDF"
else
  echo "Conversion failed"
  exit 3
fi

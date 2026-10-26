#!/usr/bin/env bash
set -euo pipefail

# Install the BPE vocabulary file required by sam3==0.1.0.
# The PyPI wheel may miss this file, but sam3's model_builder expects it at:
#   <site-packages>/assets/bpe_simple_vocab_16e6.txt.gz
#
# Preferred local layout:
#   assets/sam3/bpe_simple_vocab_16e6.txt.gz
#
# If that local file is not present, this script downloads the public OpenAI CLIP
# BPE vocabulary file.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

DST_DIR="${SITE_PACKAGES}/assets"
DST="${DST_DIR}/bpe_simple_vocab_16e6.txt.gz"
LOCAL_SRC="${PROJECT_ROOT}/assets/sam3/bpe_simple_vocab_16e6.txt.gz"
URL="https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"

mkdir -p "${DST_DIR}"

if [[ -f "${DST}" ]]; then
  echo "[SAM3 asset] already exists: ${DST}"
  exit 0
fi

if [[ -f "${LOCAL_SRC}" ]]; then
  cp "${LOCAL_SRC}" "${DST}"
  echo "[SAM3 asset] copied from repo: ${LOCAL_SRC}"
else
  echo "[SAM3 asset] local asset not found: ${LOCAL_SRC}"
  echo "[SAM3 asset] downloading from: ${URL}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${DST}" "${URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${URL}" -o "${DST}"
  else
    echo "[ERROR] neither wget nor curl is available." >&2
    echo "Place bpe_simple_vocab_16e6.txt.gz at ${LOCAL_SRC} and rerun." >&2
    exit 1
  fi
fi

ls -lh "${DST}"
python - <<'PY'
import sam3
print("sam3 import ok")
PY

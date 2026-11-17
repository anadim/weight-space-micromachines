#!/usr/bin/env bash
# BIG-Bench Hard (Suzgun et al., 2022), https://github.com/suzgunmirac/BIG-Bench-Hard, MIT License.
# Downloads the 27 task files and the official 3-shot chain-of-thought prompts at a pinned commit into data/bbh/{data,cot}/.
# The files are not redistributed in this repo; see data/bbh/LICENSE after download.
set -euo pipefail
COMMIT=9ee07bd481feebf959a6b59d61ea57bdcf30964d
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
curl -sSL "https://github.com/suzgunmirac/BIG-Bench-Hard/archive/${COMMIT}.tar.gz" | tar -xz -C "$TMP"
SRC="$TMP/BIG-Bench-Hard-${COMMIT}"
mkdir -p "$HERE/bbh/data" "$HERE/bbh/cot"
cp "$SRC"/bbh/*.json "$HERE/bbh/data/"
cp "$SRC"/cot-prompts/*.txt "$HERE/bbh/cot/"
cp "$SRC/LICENSE" "$HERE/bbh/LICENSE"
rm -rf "$TMP"
echo "BBH: $(ls "$HERE/bbh/data" | wc -l | tr -d ' ') task files, $(ls "$HERE/bbh/cot" | wc -l | tr -d ' ') prompts -> $HERE/bbh"

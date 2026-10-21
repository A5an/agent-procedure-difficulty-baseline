#!/usr/bin/env bash
# Fetch ThinkingBox-Bench v1.0 at the exact versions used for the blind test and check the dataset hashes.
set -euo pipefail
OUT="${TB_DATA:-$(cd "$(dirname "$0")/../../.." && pwd)/external_data/raw/thinkingbox}"
mkdir -p "$OUT/hf"
cd "$OUT"
[ -d thinkingbox ] || git clone -q https://github.com/microsoft/thinkingbox.git
[ -d thinkingbox-data ] || git clone -q https://github.com/microsoft/thinkingbox-data.git
git -C thinkingbox checkout -q 892964e7226044e5463ad188da119df838f4bec1
git -C thinkingbox-data checkout -q thinkingbox-bench-v1.0   # fcaba4c1a9debec42fda7f15bf29fe6d6b46c431
HF_REV=5e49f89a16c31db0b30b00950f4d654bc1708a90
for f in agents scenarios tasks; do
  [ -f "hf/$f.parquet" ] || curl -sL -o "hf/$f.parquet" \
    "https://huggingface.co/datasets/microsoft/ThinkingBox-Bench/resolve/$HF_REV/data/$f.parquet"
done
shasum -a 256 -c <<EOF
6809074441ad983b179661953d646d0627f70d7b4c534214581a2fd63d2623e6  hf/agents.parquet
850c51344efc3aa87621e95679e84bdf9f51f143b2e4a3e80ef3219da138ec54  hf/scenarios.parquet
88079e0a62bfdd69cca361165ab270503e4ff9b1719964646358966ce3223a1a  hf/tasks.parquet
EOF

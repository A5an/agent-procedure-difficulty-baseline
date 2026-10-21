#!/usr/bin/env bash
# Main run: the four agents in parallel, BATCH tasks at a time each. Logs in run/main_<agent>.log.
set -u
RUN="$(cd "$(dirname "$0")" && pwd)"
TBX="${TB_DATA:?}/thinkingbox"
cd "$TBX"
export THINKINGBOX_DATA="../thinkingbox-data" PATH="$TBX/.venv/bin:$PATH" BATCH="${BATCH:-6}"
for a in gemini-2.5-flash-lite gemini-2.5-flash gemini-2.5-pro gemini-3.8-flash; do
  /opt/miniconda3/bin/python3 "$RUN/run_tb.py" main "$a" > "$RUN/main_$a.log" 2>&1 &
done
wait
echo "main run finished"

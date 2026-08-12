#!/bin/bash
# Background snapshot daemon. Wakes every 6h; collect_snapshot.py itself skips if today's
# UTC snapshot already exists, so this accumulates ~1 distinct day per calendar day toward
# the >= 5-day target. Resumable: safe to kill and relaunch. Logs to data/snapshots/collector.log.
set -uo pipefail  # note: NOT -e — a transient collect failure must not kill the daemon.

cd "$HOME/dev/vol-lab" || { echo "[daemon] cannot cd to project dir" >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "[daemon] venv activate failed" >&2; exit 1; }

LOG="data/snapshots/collector.log"
echo "[daemon] started $(date -u +%FT%TZ) pid=$$" >> "$LOG"

while true; do
  echo "[daemon] tick $(date -u +%FT%TZ)" >> "$LOG"
  python scripts/collect_snapshot.py >> "$LOG" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    # collect_snapshot already retries transient HTTP errors; a non-zero exit here means
    # it gave up. Log it and keep looping — the next tick may succeed.
    echo "[daemon] collect exited $rc at $(date -u +%FT%TZ); continuing" >> "$LOG"
  fi
  sleep 21600  # 6 hours
done

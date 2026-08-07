#!/bin/bash
# Background snapshot daemon (ORCH, W0). Wakes every 6h; collect_snapshot.py itself
# skips if today's UTC snapshot already exists, so this accumulates ~1 distinct day
# per calendar day toward the mission's >= 5-day target. Resumable: safe to kill and
# relaunch. Logs to data/snapshots/collector.log.
cd "$HOME/dev/vol-lab" || exit 1
source .venv/bin/activate
LOG="data/snapshots/collector.log"
echo "[daemon] started $(date -u +%FT%TZ) pid=$$" >> "$LOG"
while true; do
  python scripts/collect_snapshot.py >> "$LOG" 2>&1
  sleep 21600  # 6 hours
done

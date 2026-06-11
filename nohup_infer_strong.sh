#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

if [ -f logs/infer_strong.pid ] && kill -0 "$(cat logs/infer_strong.pid)" 2>/dev/null; then
  echo "Strong inference/export is already running, pid=$(cat logs/infer_strong.pid)"
  exit 1
fi

LOG_FILE="logs/nohup_infer_strong_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_export_and_infer_strong.sh > "$LOG_FILE" 2>&1 &
echo $! > logs/infer_strong.pid

echo "started strong export/inference"
echo "pid: $(cat logs/infer_strong.pid)"
echo "log: $LOG_FILE"
echo "tail command: tail -f $LOG_FILE"

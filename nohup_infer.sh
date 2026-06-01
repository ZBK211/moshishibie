#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

if [ -f logs/infer.pid ] && kill -0 "$(cat logs/infer.pid)" 2>/dev/null; then
  echo "Inference/export is already running, pid=$(cat logs/infer.pid)"
  exit 1
fi

LOG_FILE="logs/nohup_infer_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_export_and_infer.sh > "$LOG_FILE" 2>&1 &
echo $! > logs/infer.pid

echo "started export/inference"
echo "pid: $(cat logs/infer.pid)"
echo "log: $LOG_FILE"
echo "tail command: tail -f $LOG_FILE"

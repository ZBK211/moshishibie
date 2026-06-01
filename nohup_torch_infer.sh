#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

if [ -f logs/torch_infer.pid ] && kill -0 "$(cat logs/torch_infer.pid)" 2>/dev/null; then
  echo "Torch inference is already running, pid=$(cat logs/torch_infer.pid)"
  exit 1
fi

LOG_FILE="logs/nohup_torch_infer_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_torch_infer.sh > "$LOG_FILE" 2>&1 &
echo $! > logs/torch_infer.pid

echo "started torch inference"
echo "pid: $(cat logs/torch_infer.pid)"
echo "log: $LOG_FILE"
echo "tail command: tail -f $LOG_FILE"

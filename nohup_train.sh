#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

if [ -f logs/train.pid ] && kill -0 "$(cat logs/train.pid)" 2>/dev/null; then
  echo "Training is already running, pid=$(cat logs/train.pid)"
  exit 1
fi

LOG_FILE="logs/nohup_train_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_train.sh > "$LOG_FILE" 2>&1 &
echo $! > logs/train.pid

echo "started training"
echo "pid: $(cat logs/train.pid)"
echo "log: $LOG_FILE"
echo "tail command: tail -f $LOG_FILE"

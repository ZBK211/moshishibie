#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

if [ -f logs/train_strong.pid ] && kill -0 "$(cat logs/train_strong.pid)" 2>/dev/null; then
  echo "Strong training is already running, pid=$(cat logs/train_strong.pid)"
  exit 1
fi

LOG_FILE="logs/nohup_train_strong_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_train_strong.sh > "$LOG_FILE" 2>&1 &
echo $! > logs/train_strong.pid

echo "started strong training"
echo "pid: $(cat logs/train_strong.pid)"
echo "log: $LOG_FILE"
echo "tail command: tail -f $LOG_FILE"

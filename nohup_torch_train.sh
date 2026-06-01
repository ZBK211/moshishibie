#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

if [ -f logs/torch_train.pid ] && kill -0 "$(cat logs/torch_train.pid)" 2>/dev/null; then
  echo "Torch training is already running, pid=$(cat logs/torch_train.pid)"
  exit 1
fi

LOG_FILE="logs/nohup_torch_train_$(date +%Y%m%d_%H%M%S).log"

nohup bash run_torch_train.sh > "$LOG_FILE" 2>&1 &
echo $! > logs/torch_train.pid

echo "started torch training"
echo "pid: $(cat logs/torch_train.pid)"
echo "log: $LOG_FILE"
echo "tail command: tail -f $LOG_FILE"

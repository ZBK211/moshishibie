#!/usr/bin/env bash
set -euo pipefail

echo "== processes =="
for name in train infer train_strong infer_strong torch_train torch_infer; do
  pid_file="logs/${name}.pid"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name running pid=$pid"
    else
      echo "$name not running, old pid=$pid"
    fi
  else
    echo "$name pid file not found"
  fi
done

echo
echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "== latest logs =="
ls -lh logs 2>/dev/null | tail -20 || true

echo
echo "== results =="
ls -lh results 2>/dev/null || true

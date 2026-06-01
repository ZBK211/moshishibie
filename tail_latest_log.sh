#!/usr/bin/env bash
set -euo pipefail

latest="$(ls -t logs/*.log 2>/dev/null | head -1 || true)"
if [ -z "$latest" ]; then
  echo "No log file found under logs/"
  exit 1
fi

echo "tailing $latest"
tail -f "$latest"

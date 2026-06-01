#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/train_repo data/raw/test_repo

python -m pip install -q aistudio-sdk

aistudio download --dataset 65911/SkLXRq6Q --local_dir data/raw/train_repo --max-workers 1
aistudio download --dataset 65911/51vG7A8E --local_dir data/raw/test_repo --max-workers 1

mkdir -p data
cp data/raw/train_repo/train_images.zip data/
cp data/raw/train_repo/train_label.csv data/
cp data/raw/test_repo/test_images.zip data/

ls -lh data

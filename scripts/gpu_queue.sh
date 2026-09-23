#!/usr/bin/env bash
# Sequential GPU jobs after the main models: leave-one-show-out folds, then the RF-DETR baseline.
# One job at a time (see AGENTS.md: concurrent trainings exceed GPU memory).
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."

docs() { # docs <show> <split...>: document ids of a show in the given splits
  uv run python -c "
import json,sys
from mdet.data import read_split
m=json.load(open('dataset/meta.json'))
print(' '.join(d for s in sys.argv[2:] for d in read_split(s) if m[d]['show']==sys.argv[1]))" "$@"
}

for fold in "show-a:show-a" "show-b:show-b"; do
  name="${fold%%:*}"; show="${fold#*:}"
  out="runs/loso/$name"; mkdir -p "$out"
  train=$(docs "$show" train); val=$(docs "$show" val)
  [ -f "$out/measurenet/best.pt" ] || uv run python -m mdet.train_measurenet --epochs 150 --train-docs $train --val-docs $val --out "$out/measurenet" > "$out/measurenet.log" 2>&1
  [ -f "$out/labelocr/best.pt" ] || uv run python -m mdet.train_labelocr --epochs 60 --workers 4 --train-docs $train --val-docs $val --out "$out/labelocr" > "$out/labelocr.log" 2>&1
done

if [ ! -f runs/rfdetr/model/checkpoint_best_total.pth ]; then
  (cd baselines/rfdetr && uv run python train.py --data ../../runs/rfdetr/data --out ../../runs/rfdetr/model) > runs/rfdetr/train.log 2>&1
fi
for split in valid test; do
  (cd baselines/rfdetr && uv run python predict.py --model ../../runs/rfdetr/model/checkpoint_best_total.pth --images ../../runs/rfdetr/data/$split --out ../../runs/rfdetr/pred_$split.json) >> runs/rfdetr/predict.log 2>&1
done
echo "gpu queue done"

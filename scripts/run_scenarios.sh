#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(pwd)
PYTHON=/home/vali/Desktop/ULP/shared-venv/bin/python

SCENARIOS=(
  "scenario_vbd_16k"
  "scenario_vbd_8k"
  "scenario_dns5_16k"
  "scenario_dns5_vbd16k"
)

for scenario in "${SCENARIOS[@]}"; do
  cfg="${PROJECT_ROOT}/configs/${scenario}.yaml"
  echo "=== Running scenario $scenario ==="
  $PYTHON repro.py prepare_dataset --config "$cfg" --force
  $PYTHON repro.py train_stage1 --config "$cfg" --device cuda
  $PYTHON repro.py train_qat --config "$cfg" --device cuda
  $PYTHON repro.py evaluate --config "$cfg" --device auto
  $PYTHON repro.py report --config "$cfg"
done

# cross eval
$PYTHON repro.py evaluate --config "${PROJECT_ROOT}/configs/scenario_cross_dns5_to_vbd.yaml" --device auto
$PYTHON repro.py evaluate --config "${PROJECT_ROOT}/configs/scenario_cross_vbd_to_dns5.yaml" --device auto

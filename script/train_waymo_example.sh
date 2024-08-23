#!/bin/bash
# scenes=("031" "002")
scenes=("031")
for scene in "${scenes[@]}"; do
    python train.py --config configs/example/waymo_train_$scene.yaml
done

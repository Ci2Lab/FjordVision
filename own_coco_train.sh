#!/bin/bash
cd /mnt/RAID/projects/FjordVision

# Activate virtual environment
source fjordvision/bin/activate

# Define a list of alpha values
alpha_values=(0 0.2 0.5 0.8 1)

# List of experiments to run
declare -A experiments
experiments["ablation-remove-features-coco"]="scripts/train-ablations/coco-hierarchical-classification-training.py"
experiments["ablation-remove-features-fjord"]="scripts/train-ablations/hierarchical-classification-training.py"

# Loop through each experiment
for experiment in "${!experiments[@]}"
do
    script=${experiments[$experiment]}
    echo "Running $experiment experiments:"
    # Loop through each alpha value
    for alpha_value in "${alpha_values[@]}"
    do
        echo "Running $experiment classification training with alpha = $alpha_value using script $script"
        # Run the experiment script with the alpha argument
        python3 $script --alpha $alpha_value
    done
done

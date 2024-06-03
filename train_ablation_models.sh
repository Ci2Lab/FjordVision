#!/bin/bash

# Define a list of alpha values
alpha_values=(0 0.2 0.5 0.8 1)

# List of experiments to run
declare -A experiments
experiments["rt-detr"]="scripts/rt-detr/hierarchical-classification-training.py"
experiments["yolov9"]="scripts/yolov9/hierarchical-classification-training.py"
experiments["yolov8"]="scripts/hierarchical-classification-training.py"

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

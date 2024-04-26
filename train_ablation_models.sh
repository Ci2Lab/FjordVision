#!/bin/bash
cd /mnt/RAID/projects/FjordVision

# Activate virtual environment
source fjordvision/bin/activate

# Define a list of alpha values
alpha_values=(0 0.2 0.5 0.8 1)

# List of experiments to run
declare -A experiments
experiments["attention-removed-coco"]="scripts/attention_removed/coco-hierarchical-classification-training.py"
experiments["attention-removed"]="scripts/attention_removed/hierarchical-classification-training.py"
experiments["increased-features-complexity-coco"]="scripts/increased_features_complexity/coco-hierarchical-classification-training.py"
experiments["increased-features-complexity"]="scripts/increased_features_complexity/hierarchical-classification-training.py"
experiments["decreased-branch-complexity-coco"]="scripts/decreased_branch_complexity/coco-hierarchical-classification-training.py"
experiments["decreased-branch-complexity"]="scripts/decreased_branch_complexity/hierarchical-classification-training.py"
experiments["remove-features-coco"]="scripts/remove_features/coco-hierarchical-classification-training.py"
experiments["remove-features"]="scripts/remove_features/hierarchical-classification-training.py"

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

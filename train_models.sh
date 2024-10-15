#!/bin/bash

# Define a list of alpha values
alpha_values=(0 0.2 0.5 0.8 1)

# List of experiments to run
declare -A experiments
experiments["yolov8"]="scripts/hierarchical-classification-training.py"
experiments["yolov9"]="scripts/yolov9/hierarchical-classification-training.py"
experiments["rtdetr"]="scripts/rt-detr/hierarchical-classification-training.py"
experiments["maskrcnn"]="scripts/mask-rcnn/hierarchical-classification-training.py"
experiments["attention_removed"]="scripts/attention_removed/hierarchical-classification-training.py"
experiments["decreased_branch_complexity"]="scripts/decreased_branch_complexity/hierarchical-classification-training.py"
experiments["increased_features_complexity"]="scripts/increased_features_complexity/hierarchical-classification-training.py"
experiments["remove_features"]="scripts/remove_features/hierarchical-classification-training.py"

# Log file to capture the output and errors
log_file="experiment_log.txt"
echo "Experiment started at $(date)" > $log_file

# Loop through each experiment
for experiment in "${!experiments[@]}"
do
    script=${experiments[$experiment]}
    echo "Running $experiment experiments:" | tee -a $log_file
    # Loop through each alpha value
    for alpha_value in "${alpha_values[@]}"
    do
        echo "Running $experiment classification training with alpha = $alpha_value using script $script" | tee -a $log_file
        # Run the experiment script with the alpha argument
        python3 $script --alpha $alpha_value >> $log_file 2>&1
        if [ $? -ne 0 ]; then
            echo "Error running $experiment classification training with alpha = $alpha_value" | tee -a $log_file
        fi
    done
done

echo "Experiment finished at $(date)" | tee -a $log_file

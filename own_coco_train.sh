#!/bin/bash
cd /mnt/RAID/projects/FjordVision

# Activate venv
source fjordvision/bin/activate

# Define a list of alpha values
alpha_values=(0.5)

# Loop through each alpha value
for alpha_value in "${alpha_values[@]}"
do
    echo "Running hierarchical classification training with alpha = $alpha_value"
    
    # Run the Python script with the alpha argument
    python3 scripts/hierarchical-classification-training.py --alpha $alpha_value
done

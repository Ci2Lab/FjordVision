#!/bin/bash
cd "$(dirname "$0")"

# Activate virtual environment
source fjordvision/bin/activate

# Setup base dataset directory
mkdir -p datasets  # Ensure the datasets directory exists

# Download and unzip the comprehensive datasets and models from Figshare
echo "Downloading and preparing all necessary datasets and models..."
wget -O datasets/datasets.zip 'https://figshare.com/ndownloader/files/xxxxxxx'  # Replace xxxxxxx with actual file ID
unzip -o datasets/datasets.zip -d datasets
rm datasets/datasets.zip

# Check if the Fjord Dataset is prepared
if [ ! -f "datasets/The Fjord Dataset/fjord.yaml" ]; then
    echo "Error: Fjord dataset not found after extraction."
    exit 1
else
    echo "Fjord dataset is successfully prepared."
fi

# Check if the hierarchical model weights and other datasets are prepared
if [ ! -d "datasets/hierarchical-model-weights" ] || [ ! -d "datasets/pre-trained-models/coco" ]; then
    echo "Error: Required model weights or datasets are missing."
    exit 1
else
    echo "Hierarchical model weights and pre-trained models are successfully prepared."
fi

# Optionally, execute scripts for data processing and training based on the prepared datasets
echo "Running additional data processing and training scripts..."
python3 scripts/coco-dataset-creation-postprocess.py
python3 scripts/coco-hierarchical-classification-training.py
python3 scripts/dataset-creation-postprocess.py
python3 scripts/hierarchical-classification-training.py

echo "All processes completed successfully."

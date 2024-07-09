#!/bin/bash
# Navigate to the project directory
cd "$(dirname "$0")"

# Create and activate a virtual environment
python3 -m venv fjordvision
source fjordvision/bin/activate

# Install required dependencies
pip install -r requirements.txt

echo "Setup complete. Environment is ready."

#!/bin/bash
cd "$(dirname "$0")"

# Ensure the base dataset directory exists
mkdir -p datasets

# Download and unzip 'Esefjorden Marine Vegetation Segmentation Dataset' from Figshare
if [ ! -f "datasets/EMVSD/EMVSD.yaml" ]; then
    echo "Downloading 'Esefjorden Marine Vegetation Segmentation Dataset'..."
    wget -O EMVSD.zip 'https://figshare.com/ndownloader/files/47516684'
    unzip -o EMVSD.zip -d datasets
    rm EMVSD.zip
    echo "'Esefjorden Marine Vegetation Segmentation Dataset' is ready."
else
    echo "'Esefjorden Marine Vegetation Segmentation Dataset' already exists."
fi

# Download and unzip 'FjordVision Experimental Data' from Figshare
if [ ! -d "datasets/hierarchical-model-weights" ]; then
    echo "Downloading 'FjordVision Experimental Data'..."
    wget -O experimental_data.zip 'https://figshare.com/ndownloader/files/47514674'
    unzip -o experimental_data.zip -d datasets  # Extract directly to the datasets directory
    rm experimental_data.zip
    echo "'FjordVision Experimental Data' is ready."
else
    echo "'FjordVision Experimental Data' already exists."
fi

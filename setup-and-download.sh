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
cd "$(dirname "$0")"c

# Ensure the base dataset directory exists
mkdir -p datasets

# Download and unzip 'The Fjord Dataset' from Figshare
if [ ! -f "datasets/The Fjord Dataset/fjord.yaml" ]; then
    echo "Downloading 'The Fjord Dataset'..."
    wget -O fjord_dataset.zip 'https://figshare.com/ndownloader/files/45433267'
    unzip -o fjord_dataset.zip -d datasets
    rm fjord_dataset.zip
    echo "'The Fjord Dataset' is ready."
else
    echo "'The Fjord Dataset' already exists."
fi

# Download and unzip 'FjordVision Experimental Data' from Figshare
if [ ! -d "datasets/hierarchical-model-weights" ] || [ ! -d "datasets/pre-trained-models/coco" ]; then
    echo "Downloading 'FjordVision Experimental Data'..."
    wget -O experimental_data.zip 'https://figshare.com/ndownloader/files/45875268'
    unzip -o experimental_data.zip -d .  # Extract directly to the current directory
    rm experimental_data.zip
    echo "'FjordVision Experimental Data' is ready."
else
    echo "'FjordVision Experimental Data' already exists."
fi

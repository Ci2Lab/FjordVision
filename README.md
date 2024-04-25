# FjordVision Hierarchical Classification Models

## Description
The FjordVision project explores the impact of various alpha regularization parameters on the performance of hierarchical classification models using a self developed marine dataset and the COCO dataset. This project aims to determine how model configurations affect the recognition capabilities for categories such as marine species and vehicles in the COCO dataset.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Experiments and Reproducibility](#experiments-and-reproducibility)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
To set up your environment for running these experiments, follow these steps:
```bash
# Clone the repository
git clone https://github.com/bcwein/FjordVision.git

# Navigate to the project directory
cd FjordVision

# Create and activate a virtual environment
python -m venv fjordvision
source fjordvision/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

## Usage

This section outlines the necessary steps to utilize the project for training new models as well as testing with pre-trained models once available.

### Train and Load YOLO V8 Models

#### Download and Prepare the Dataset

The Fjord Dataset is crucial for training the YOLO V8 models. The script provided automates the process of downloading, unzipping, and preparing the dataset for training, ensuring all necessary files are correctly set up.

#### Training the Models

To train YOLO V8 models using the prepared Fjord Dataset, follow these simple steps:

1. **Download and Prepare the Dataset**: Ensure the dataset is ready by running the script which handles dataset preparation and training initiation:

    ```bash
    ./train-yolov8.sh
    ```

    This script performs the following actions:
    - Checks if the dataset is already present.
    - Downloads the dataset ZIP file if it's not present.
    - Unzips the dataset to the correct directory.
    - Checks if the `fjord.yaml` is in the correct location.
    - Initiates the training process using the dataset.

2. **Running the Training Script**: The above command will automatically continue to train the models using the YOLO V8 configuration as specified in the Python script located at `scripts/train-yolov8.py`.

    Detailed logs and outputs from the training process will be saved in designated directories, allowing you to monitor the progress and results of the training.

**Training will take a substantial time depending on your hardware**. To start using already trained models. Consider using pre-trained models instead described in the next section.

### Download Trained Models and Datasets

This section will provide information on how you can obtain our trained models and datasets for testing our models on video streams or further analysis.

**Note:** *(Future section to be expanded upon availability of download links for models and additional datasets.)*

### Using Pre-trained Models

Once pre-trained models are available, this section will guide you on how to download and use them, including detailed instructions for deploying the models for inference on new data.


# FjordVision Hierarchical Classification Model

This github repo is the code implementation of our paper
*Marine Vegetation Analysis in Esefjorden, Norway using Hierarchical CNN-based Classification*.

## Demo: Taxonomic Classification at Different Levels

Explore our hierarchical classification models across various taxonomic ranks:

### Species Level
![Species GIF](demo/output-species.gif)

### Class level
![Species GIF](demo/output-class.gif)

### Binary level
![Species GIF](demo/output-binary.gif)

## Description
The FjordVision project explores the impact of various alpha regularization parameters on the performance of hierarchical classification models using a self developed marine dataset and the COCO dataset. This project aims to determine how model configurations affect the recognition capabilities for categories such as marine species and vehicles in the COCO dataset.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Detailed Documentation](#detailed-documentation)

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

This section outlines the necessary steps to utilize the project for running our code with pre-trained models and experimental data.

### Download datasets and experimental data

Run the script

```bash
./setup-and-download.sh
```
To download the datasets and experimental data that the rest of code uses.
This will download several things.

- **Esefjorden Marine Vegetation Segmentation Dataset (EMVSD)** which contains 17000 annotated images
in the [Yolo txt format](https://docs.ultralytics.com/datasets/segment/) and COCO mask format.
- Weights of Yolo v8, Yolo v9 and RTDETR and Mask-RCNN for EMVSD.
- Datasets used for training our hierarcical classifiers. These are stored as a 
  combination of *parquet* containing labels and folders of segmented images.
- Weights of trained hierarcical classification models. 

### Run our model on video

If you want to run inference with our models on a video stream use the following
command

```bash
python3 scripts/run_classification_stream.py --display_level="species" --output_path="demo/output-species.mp4"
```

Available hierarcical levels are "species", "genus", "class", "binary".

### Run experiments and gather F1 scores

This is done by running the notebook `run-experiments.ipyn`

### Visualise activations and class activiations

This is done by running the notebook `hierarcical-cnn-visualisation.ipyn`

## Project Structure

- **datasets/**: Contains the EMVSD, including segmented images and model weights.
- **demo/**: Demo videos and GIFs for visualizing model outputs.
- **fjordvision/**: Python virtual enviromnent
- **models/**: Definitions of the hierarchical models.
- **preprocessing/**: Scripts for data preprocessing, dataset creation and model training.
- **scripts/**: Utility scripts for setting up the environment, downloading data, and running the models.
- **utils/**: Helper functions and utilities for model operation and data manipulation.
- **experimental-data.ipynb**: Jupyter notebook for initial data analysis.
- **hierarchical-cnn-visualisation.ipynb**, **run-experiments.ipynb**: Notebooks for running experiments and visualizing model performance.

## Detailed Documentation

For more in-depth documentation of the modules and code, see the following links:
- [BranchCNN Model Documentation](docs/branch_cnn.md)
- [BranchCNN Model Documentation](docs/hierarcical_cnn.md)

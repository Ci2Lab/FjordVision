import os
from pathlib import Path
import yaml
from ultralytics import YOLO, RTDETR

# Convert dataset YAML path to absolute path
dataset_yaml_path = Path('datasets/EMVSD/EMVSD.yaml').resolve()
dataset_images_path = Path('datasets/EMVSD/EMVSD').resolve()

# Load and update the dataset YAML file with absolute paths
with open(dataset_yaml_path, 'r') as file:
    data_config = yaml.safe_load(file)

# Update paths to absolute paths
data_config['path'] = str(dataset_images_path)
data_config['train'] = str((dataset_images_path / 'images/train').resolve())
data_config['val'] = str((dataset_images_path / 'images/val').resolve())
data_config['test'] = str((dataset_images_path / 'images/test').resolve())

# Ensure the correct format and class information
data_config['nc'] = 13
data_config['names'] = [
    'asterias rubens', 'asteroidea', 'fucus vesiculosus', 'henricia', 
    'mytilus edulis', 'myxine glurinosa', 'pipe', 'rock', 
    'saccharina latissima', 'tree', 'ulva intestinalis', 'urospora', 'zostera marina'
]

# Save the updated YAML file
updated_dataset_yaml_path = dataset_yaml_path.parent / 'EMVSD_abs.yaml'
with open(updated_dataset_yaml_path, 'w') as file:
    yaml.safe_dump(data_config, file)

# Model configurations with corresponding YAML files
models = [
    ("rtdetr-l", "rtdetr-l.yaml"),
    ("yolov10n", "yolov10n.yaml")
]

# Function to determine if a model needs training
def needs_training(experiment_name):
    return not os.path.exists(f"runs/segment/{experiment_name}")

# Process each model configuration
for model_name, yaml_file in models:
    experiment_name = f"{model_name}-seg-train"

    # Check if this experiment already exists
    if needs_training(experiment_name):
        # Select appropriate model class
        if model_name.startswith("yolov8") or model_name.startswith("yolov10"):
            model = YOLO(yaml_file)
        elif model_name.startswith("rtdetr"):
            model = RTDETR(yaml_file)
        else:
            continue

        epochs = 100  # Adjusted epochs for demonstration

        # Train the model
        model.train(
            data=str(updated_dataset_yaml_path),
            epochs=epochs,
            name=experiment_name
        )

        # Evaluate model performance
        metrics = model.val()

from ultralytics import YOLO
import os.path.exists as exists

# List of model configurations and their respective sizes
models = [
    ("yolov8n-seg.yaml", "n"),
    ("yolov8s-seg.yaml", "s"),
    ("yolov8l-seg.yaml", "l"),
    ("yolov8x-seg.yaml", "x")
]

# Iterate over each model configuration
for config, size in models:

    # Descriptive experiment name for fine-tuning
    experiment_name = f"Yolov8{size}-seg-finetune"

    if exists(experiment_name):
        continue

    # Load the model with random weights
    model = YOLO(config)

    # Train the model
    model.train(
        data="/mnt/RAID/datasets/The Fjord Dataset/fjord.yaml",
        batch=-1,
        epochs=600,
        name=experiment_name  # Set experiment name
    )

    # Evaluate model performance
    metrics = model.val()

from ultralytics import YOLO
import os
# List of pretrained model weights
models = [
    ("yolov8n-seg.pt", "n"),
    ("yolov8s-seg.pt", "s"),
    ("yolov8m-seg.pt", "m"),
    ("yolov8l-seg.pt", "l"),
    ("yolov8x-seg.pt", "x")
]

# Iterate over each pretrained model
for weights, size in models:
    # Descriptive experiment name for fine-tuning
    experiment_name = f"Yolov8{size}-seg-finetune"

    if os.path.exists("runs/segment" + experiment_name):
        continue

    # Load the pretrained model
    model = YOLO(weights)

    # Train (fine-tune) the model
    model.train(
        data="/mnt/RAID/datasets/The Fjord Dataset/fjord.yaml",
        batch=-1,
        epochs=300,
        name=experiment_name  # Set experiment name
    )

    # Evaluate model performance
    metrics = model.val()

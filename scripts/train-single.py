from ultralytics import YOLO

# List of model configurations and their respective sizes
models = [
    ("yolov8l-seg.yaml", "l"),
]

# Iterate over each model configuration
for config, size in models:

    # Descriptive experiment name for fine-tuning
    experiment_name = f"Yolov8{size}-seg-finetune"

    # Load the model with random weights
    model = YOLO("/mnt/RAID/projects/FjordVision/runs/segment/Yolov8l-seg-train/weights/last.pt")

    # Train the model
    model.train(
        data="/mnt/RAID/datasets/The Fjord Dataset/fjord.yaml",
        batch=-1,
        resume=True
    )

    # Evaluate model performance
    metrics = model.val()

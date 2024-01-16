from ultralytics import YOLO

# List of model configurations and their respective sizes
models = [
    ("yolov8n-seg.yaml", "n"),
    ("yolov8s-seg.yaml", "s"),
    ("yolov8l-seg.yaml", "l"),
    ("yolov8x-seg.yaml", "x")
]

# Iterate over each model configuration
for config, size in models:
    # Load the model with random weights
    model = YOLO(config)

    # Descriptive experiment name
    experiment_name = f"Yolov8{size}Z-seg-train"

    # Train the model
    model.train(
        data="/mnt/RAID/datasets/The Fjord Dataset/fjord.yaml",
        batch=-1,
        epochs=600,
        name=experiment_name  # Set experiment name
    )

    # Evaluate model performance
    metrics = model.val()

from ultralytics import YOLO
import os

# Model configurations with corresponding YAML files
models = [
    ("yolov8n", "yolov8n.yaml"),
    ("yolov8s", "yolov8s.yaml"),
    ("yolov8m", "yolov8m.yaml"),
    ("yolov8l", "yolov8l.yaml"),
    ("yolov8x", "yolov8x.yaml")
]

# Function to determine if a model needs training
def needs_training(experiment_name):
    return not os.path.exists(f"runs/segment/{experiment_name}")

# Process each model configuration
for model_name, yaml_file in models:
    experiment_name = f"{model_name}-seg-train"

    # Check if this experiment already exists
    if needs_training(experiment_name):
        # Build and train a new model from YAML
        model = YOLO(yaml_file)
        epochs = 600  # Set epochs for training from scratch

        # Train the model
        model.train(
            data="datasets/The Fjord Dataset/fjord.yaml",
            epochs=epochs,
            name=experiment_name
        )

        # Evaluate model performance
        metrics = model.val()

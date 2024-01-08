from ultralytics import YOLO

model = YOLO("yolov8m-seg.yaml")  # load a model with random weights

# Use the model
model.train(
    data="/mnt/RAID/datasets/The Fjord Dataset/fjord3.yaml",
    batch=30,
    epochs=600)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
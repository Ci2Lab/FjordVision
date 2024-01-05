from ultralytics import YOLO

model = YOLO("yolov8m-seg.yaml")  # load a pretrained model (recommended for training)

# Use the model
model.train(
    data="/mnt/RAID/datasets/The Fjord Dataset/fjord3-species.yaml",
    batch=-1,
    workers=20,
    epochs=300)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
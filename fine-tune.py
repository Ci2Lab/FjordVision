from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")  # load a pretrained model

# Use the model
model.train(
    data="/mnt/RAID/datasets/The Fjord Dataset/fjord3-species.yaml",
    batch=-1,
    workers=20)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
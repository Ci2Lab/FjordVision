import os
import sys
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import numpy as np

# Add the project directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.engine import evaluate, train_one_epoch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.ids = [img_id for img_id in self.ids if self._has_valid_annotation(img_id)]

    def _has_valid_annotation(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
            if w > 0 and h > 0:
                mask = self.coco.annToMask(ann)
                if np.any(mask[ymin:ymin + h, xmin:xmin + w]):
                    return True
        return False

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')

        num_objs = len(anns)

        boxes = []
        masks = []
        labels = []
        for i in range(num_objs):
            xmin, ymin, w, h = anns[i]['bbox']
            xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
            if w <= 0 or h <= 0:
                continue
            xmax = xmin + w
            ymax = ymin + h
            if xmax > xmin and ymax > ymin:
                mask = coco.annToMask(anns[i])
                if np.any(mask[ymin:ymax, xmin:xmax]):
                    boxes.append([xmin, ymin, xmax, ymax])
                    masks.append(mask)
                    labels.append(anns[i]['category_id'])

        if not boxes:
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks])

        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='COCO_V1')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def load_classes(filepath):
    with open(filepath, 'r') as f:
        classes = f.read().strip().split('\n')
    return classes

def save_initial_model():
    classes = load_classes('datasets/EMVSD/EMVSD/classes.txt')
    num_classes = len(classes) + 1  # add 1 for background class

    model = get_model_instance_segmentation(num_classes)
    model_path = 'models/model_best.pth'
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    print(f"Initial model with pretrained weights saved at {model_path}.")

def evaluate_model_on_test_set():
    classes = load_classes('datasets/EMVSD/EMVSD/classes.txt')
    num_classes = len(classes) + 1  # add 1 for background class
    batch_size = 10  # Set the desired batch size here

    dataset_test = CocoDataset('datasets/EMVSD/EMVSD', 'datasets/EMVSD/EMVSD/val_annotations.json', get_transform(train=False))

    data_loader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    model = get_model_instance_segmentation(num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    checkpoint_path = 'models/model_best.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        print("Evaluating model on the test dataset...")
        evaluate(model, data_loader_test, device=device)
    else:
        print(f"Checkpoint {checkpoint_path} not found. Please train the model first.")
        sys.exit(1)

def train_model():
    classes = load_classes('datasets/EMVSD/EMVSD/classes.txt')
    num_classes = len(classes) + 1  # add 1 for background class
    batch_size = 10  # Set the desired batch size here

    dataset = CocoDataset('datasets/EMVSD/EMVSD', 'datasets/EMVSD/EMVSD/train_annotations.json', get_transform(train=True))
    dataset_test = CocoDataset('datasets/EMVSD/EMVSD', 'datasets/EMVSD/EMVSD/val_annotations.json', get_transform(train=False))

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    model = get_model_instance_segmentation(num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 100
    early_stopping_patience = 10
    best_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, data_loader_test, device=device)

        if coco_evaluator is not None:
            val_loss = coco_evaluator.coco_eval['bbox'].stats[0]
        else:
            print("Warning: Evaluation returned None.")
            val_loss = float('inf')

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} took {epoch_time / 60:.2f} minutes with validation loss {val_loss:.4f}")

        save_model(model, optimizer, epoch, 'datasets/MaskRCNN-weights/model_last.pth')

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, optimizer, epoch, 'datasets/MaskRCNN-weights/model_best.pth')
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")

def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and Evaluate Mask R-CNN model")
    parser.add_argument('--test', action='store_true', help='Run in test mode for immediate evaluation')
    parser.add_argument('--save-initial', action='store_true', help='Save initial model with pretrained weights')
    args = parser.parse_args()

    if args.save_initial:
        save_initial_model()
    elif args.test:
        evaluate_model_on_test_set()
    else:
        train_model()

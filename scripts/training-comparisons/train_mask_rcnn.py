import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        num_objs = len(anns)

        boxes = []
        labels = []
        masks = []

        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])
            masks.append(coco.annToMask(anns[i]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            img = self.apply_transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

    def apply_transforms(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img

def get_transform():
    transforms = []
    transforms.append(F.to_tensor)
    return transforms

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    dataset = COCODataset(root='datasets/EMVSD/EMVSD/images/train', annFile='datasets/EMVSD/EMVSD/train_annotations.json', transforms=get_transform())
    dataset_val = COCODataset(root='datasets/EMVSD/EMVSD/images/val', annFile='datasets/EMVSD/EMVSD/val_annotations.json', transforms=get_transform())

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load pre-trained Mask R-CNN weights and modify for custom number of classes
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 14)  # 13 classes + background

    model = model.to('cuda')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 100
    early_stopping_patience = 10
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            images = list(image.to('cuda') for image in images)
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        for images, targets in data_loader_val:
            images = list(image.to('cuda') for image in images)
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(data_loader_val)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()

import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
import kagglehub

# Download the dataset
path = kagglehub.dataset_download("kailaspsudheer/sarscope-unveiling-the-maritime-landscape")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class COCODataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.ann_file = os.path.join(root, '_annotations.coco.json')
        self.coco = COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Handle no annotations
        if boxes.numel() == 0:
            boxes = boxes.view(0, 4)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        image_id = torch.tensor([img_id])
        if boxes.shape[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target
        
    def __len__(self):
        return len(self.ids)

def get_transform(train):
    transforms_list = [T.ToTensor()]
    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_list)

train_dataset = COCODataset(root=os.path.join(path, 'train'), transforms=get_transform(train=True))
valid_dataset = COCODataset(root=os.path.join(path, 'valid'), transforms=get_transform(train=False))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=lambda batch: tuple(zip(*batch))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    collate_fn=lambda batch: tuple(zip(*batch))
)

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_model(num_classes):
    # Use recommended weights parameter instead of pretrained
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=2)  # background + ship
model.to(device)

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Resume from checkpoint if available
start_epoch = 0
checkpoint_path = "ship_detector_checkpoint.pth"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint to resume training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed from epoch {start_epoch}")

num_epochs = 10
for epoch in range(start_epoch, num_epochs):
    model.train()
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # If no valid targets, skip
        if len(targets) == 0 or any(len(t["boxes"]) == 0 for t in targets):
            print("Warning: no valid targets for this batch. Skipping.")
            continue

        loss_dict = model(imgs, targets)
        
        # Ensure loss_dict is a dictionary
        if not isinstance(loss_dict, dict):
            print("Warning: model returned non-dict output in training mode. Skipping.")
            continue
        
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_loader)}] Loss: {losses.item():.4f}")
    
    lr_scheduler.step()
    
    # Validation
    model.eval()
    val_loss = 0
    count_val_batches = 0
    with torch.no_grad():
        for imgs, targets in valid_loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            if len(targets) == 0 or any(len(t["boxes"]) == 0 for t in targets):
                continue
            
            loss_dict = model(imgs, targets)
            if not isinstance(loss_dict, dict):
                continue
            
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()
            count_val_batches += 1
    
    if count_val_batches > 0:
        val_loss /= count_val_batches
    else:
        val_loss = float('inf')  # If no valid validation batches
    
    print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }, checkpoint_path)

# Test inference
test_dataset = COCODataset(root=os.path.join(path, 'test'), transforms=get_transform(train=False))
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=lambda batch: tuple(zip(*batch))
)

model.eval()
for imgs, targets in test_loader:
    imgs = list(img.to(device) for img in imgs)
    with torch.no_grad():
        predictions = model(imgs)
    for pred in predictions:
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        keep = scores > 0.5
        boxes = boxes[keep]
        ship_count = len(boxes)
        print(f"Detected {ship_count} ships.")

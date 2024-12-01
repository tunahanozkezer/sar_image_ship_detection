import kagglehub
import json
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
import os
from PIL import Image
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Download latest version of the dataset using kagglehub to the project directory
try:
    path = kagglehub.dataset_download("kailaspsudheer/sarscope-unveiling-the-maritime-landscape")
    print("Path to dataset files:", path)
except kagglehub.exceptions.KaggleApiHTTPError as e:
    print("Failed to download dataset:", e)
    exit(1)

# Function to create the ShipDataset class for different dataset splits
def get_ship_dataset(root, split, transforms=None):
    class ShipDataset(Dataset):
        def __init__(self, root, split, transforms=None):
            self.root = os.path.join(root, split)
            self.transforms = transforms
            self.json_file = os.path.join(self.root, "_annotations.coco.json")
            # Load the annotation file
            if not os.path.exists(self.json_file):
                raise FileNotFoundError(f"Annotation file not found: {self.json_file}")
            with open(self.json_file) as f:
                self.data = json.load(f)
            self.images = self.data['images']
            self.annotations = self.data['annotations']

        def __getitem__(self, idx):
            # Load image info
            img_info = self.images[idx]
            img_path = os.path.join(self.root, img_info["file_name"])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img = Image.open(img_path).convert("RGB")

            # Get annotations for the image
            img_id = img_info['id']
            boxes = []
            labels = []
            for ann in self.annotations:
                if ann['image_id'] == img_id:
                    # Convert COCO format (x, y, width, height) to (x_min, y_min, x_max, y_max)
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])

            # Convert to PyTorch tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}

            # Apply transformations to the image if provided
            if self.transforms is not None:
                img = self.transforms(img)

            return img, target

        def __len__(self):
            return len(self.images)

    return ShipDataset(root, split, transforms)

if __name__ == "__main__":
    # Set transforms for data
    transform = T.Compose([
        T.ToTensor(),  # Convert PIL image to tensor
    ])

    # Load dataset and dataloader
    root = path  # Use the downloaded dataset path
    # Create dataset instances for training, validation, and testing
    try:
        train_dataset = get_ship_dataset(root, split="train", transforms=transform)
        valid_dataset = get_ship_dataset(root, split="valid", transforms=transform)
        test_dataset = get_ship_dataset(root, split="test", transforms=transform)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # DataLoader for batching, shuffling, and parallel loading
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    # Model Definition
    # Load a pre-trained ResNet50 backbone and remove the last layers to use it as a feature extractor
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048  # Define the number of output channels from the backbone

    # Define the anchor generator for the RPN (Region Proposal Network)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  # Anchor sizes
        aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Aspect ratios for each anchor size
    )

    # Define the ROI (Region of Interest) pooling layer
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],  # Feature maps to use
        output_size=7,  # Size of the output after pooling
        sampling_ratio=2  # Sampling ratio for ROIAlign
    )

    # Define the Faster R-CNN model with the modified backbone and the ROI pooler
    model = FasterRCNN(
        backbone,
        num_classes=2,  # Number of classes (background + ship)
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Training Loop
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        i = 0
        for images, targets in train_dataloader:
            # Move images and targets to the appropriate device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and compute losses
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory. Skipping batch.")
                torch.cuda.empty_cache()
                continue

            # Backpropagation
            optimizer.zero_grad()  # Zero the gradients
            losses.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            i += 1
            # Print the loss for the current iteration
            print(f"Epoch [{epoch + 1}/{num_epochs}], Iter [{i}/{len(train_dataloader)}], Loss: {losses.item()} ")

    print("Training completed.")

    # Save the trained model
    torch.save(model.state_dict(), "ship_detector.pth")
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw
import numpy as np

# Path to your checkpoint
checkpoint_path = "ship_detector_checkpoint.pth"

def get_model(num_classes):
    # Use the same weights as training to keep consistency
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=2) # background + ship
model.eval()

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])

def transform_image(img):
    from torchvision.transforms import ToTensor
    return ToTensor()(img)

# Load your test image
image_path = "000017_jpg.rf.efed9f8650b726a3088904b5e4ffc832.jpg" # replace with your image path
img = Image.open(image_path).convert("RGB")

# Transform and run inference
img_tensor = transform_image(img)
with torch.no_grad():
    predictions = model([img_tensor])

pred = predictions[0]
boxes = pred['boxes'].cpu().numpy()
scores = pred['scores'].cpu().numpy()

# Confidence threshold
threshold = 0.5
keep = scores > threshold
boxes = boxes[keep]

# Count ships
ship_count = len(boxes)
print(f"Detected {ship_count} ships.")

# Draw bounding boxes on the image
draw = ImageDraw.Draw(img)
for box in boxes:
    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

# Show or save the result
img.show()
# img.save("detected_image.jpg")  # Uncomment to save the image

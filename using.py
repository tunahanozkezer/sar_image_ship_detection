import os
import json
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
import sys

# Path to your checkpoint
checkpoint_path = "ship_detector_checkpoint.pth"

def get_model(num_classes):
    # Use the same weights as training to keep consistency
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# Load the model
model = get_model(num_classes=2) # background + ship
model.eval()

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

def transform_image(img):
    return ToTensor()(img)

def process_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
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

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        return img, ship_count
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    output_dir = f"{os.path.basename(directory)}_proceed"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(directory, filename)
            print(f"Processing {filename}...")

            processed_image, ship_count = process_image(image_path)

            if processed_image:
                # Save processed image
                processed_image.save(os.path.join(output_dir, filename))

            # Save result
            results[filename] = {"ship_count": ship_count}

    # Save results to JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Processing complete. Results saved to {json_path} and images saved to {output_dir}.")

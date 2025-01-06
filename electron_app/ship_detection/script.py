import os
import json
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
import sys
import time

checkpoint_path = "ship_detection/ship_detector_checkpoint.pth"

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=2)  # background + ship
model.eval()

# Eğer checkpoint'inizi load etmeniz gerekiyorsa:
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'), weights_only=True)
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

        threshold = 0.5
        keep = scores > threshold
        boxes = boxes[keep]

        ship_count = len(boxes)

        draw = ImageDraw.Draw(img)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        return img, ship_count
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, 0

def main(directory):
    output_dir = f"{os.path.basename(directory)}_proceed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    results = {}
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(directory, filename)

            processed_image, ship_count = process_image(image_path)
            if processed_image:
                processed_image.save(os.path.join(output_dir, filename))
            results[filename] = {"val": ship_count}
    end_time = time.time()
    elapsed_time = end_time - start_time
    results["Gecen_Sure"] = {"val": elapsed_time}

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    
    
    print(results)
    # print(f"Veri isleme tamamlandi. Sonuclar {json_path}, islenmis gorsellerde {output_dir} klasorune eklendi.")
    # print(f"Toplam calisma suresi: {elapsed_time:.2f}sn.")

    # Sonucu JSON formatta da konsola (stdout) yazabiliriz ki Electron oradan parse edebilsin.
    return {
        "output_dir": output_dir,
        "results_file": json_path,
        "results": results,
        "elapsed_time": elapsed_time
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    main(directory)

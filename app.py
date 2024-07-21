import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

# Define constants
BOX_COLORS = {
    "unchecked": (242, 48, 48),
    "checked": (38, 115, 101),
    "block": (242, 159, 5)
}
BOX_PADDING = 2

# Load models
DETECTION_MODEL = YOLO("models/detector-model.pt")
CLASSIFICATION_MODEL = YOLO("models/classifier-model.pt")  # 0: block, 1: checked, 2: unchecked

# def detect(image_path):
#     """
#     Output list of results with bounding boxes and class labels

#     Args:
#     - image_path: Path to the image to check for checkboxes

#     Return: List of results with bounding boxes and class labels
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Image not found.")
#         return []

#     # Predict on image
#     results = DETECTION_MODEL.predict(source=image, conf=0.2, iou=0.8)  # Predict on image
#     boxes = results[0].boxes  # Get bounding boxes

#     result_list = []

#     if len(boxes) == 0:
#         print("No boxes detected.")
#         return result_list

#     # Get bounding boxes
#     for box in boxes:
#         detection_class_conf = round(box.conf.item(), 2)
#         detection_class = list(BOX_COLORS)[int(box.cls)]
#         # Get start and end points of the current box
#         start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
#         end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
#         box_image = image[start_box[1]:end_box[1], start_box[0]:end_box[0], :]

#         # Convert box_image to RGB format and then to PIL Image for classification model
#         box_image_rgb = cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB)
#         # pil_image = Image.fromarray(box_image_rgb)
#         pil_image = cv2.resize(box_image_rgb, (640, 640))

#         # Determine the class of the box using classification model
#         tensor_image = transforms.ToTensor()(pil_image)  # Use transforms
#         cls_results = CLASSIFICATION_MODEL.predict(source=tensor_image, conf=0.5)
#         # cls_results = CLASSIFICATION_MODEL.predict(source=pil_image, conf=0.5)
#         probs = cls_results[0].probs  # cls prob, (num_class, )
#         classification_class = list(BOX_COLORS)[2 - int(probs.top1)]
#         classification_class_conf = round(probs.top1conf.item(), 2)

#         cls = classification_class if classification_class_conf > 0.9 else detection_class

#         result_list.append({
#             "bbox": [start_box, end_box],
#             "class": cls,
#             "confidence": detection_class_conf
#         })

#     # Print the results to the console
#     for result in result_list:
#         # print(result_list)
#         print(f"Bounding Box: {result['bbox']}, Class: {result['class']}, Confidence: {result['confidence']}")

#     return result_list

def detect(image_path):
    """
    Output list of results with bounding boxes and class labels

    Args:
    - image_path: Path to the image to check for checkboxes

    Return: List of results with bounding boxes and class labels
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return []

    # Predict on image
    results = DETECTION_MODEL.predict(source=image, conf=0.2, iou=0.8)  # Predict on image
    boxes = results[0].boxes  # Get bounding boxes

    result_list = []

    if len(boxes) == 0:
        print("No boxes detected.")
        return result_list

    # Define transforms
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    # Get bounding boxes
    for box in boxes:
        detection_class_conf = round(box.conf.item(), 2)
        detection_class = list(BOX_COLORS.keys())[int(box.cls)]
        # Get start and end points of the current box
        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
        box_image = image[start_box[1]:end_box[1], start_box[0]:end_box[0], :]

        # Convert box_image to RGB format
        box_image_rgb = cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image and then apply transforms
        # pil_image = Image.fromarray(box_image_rgb)
        pil_image = cv2.resize(box_image_rgb, (640, 640))
        tensor_image = preprocess(pil_image)  # Use transforms

        # Determine the class of the box using classification model
        cls_results = CLASSIFICATION_MODEL.predict(source=tensor_image, conf=0.5)
        probs = cls_results[0].probs  # cls prob, (num_class, )

        # Print debug information
        print(f"Bounding Box: {start_box} to {end_box}")
        print(f"Raw classification probabilities: {probs}")

        classification_class_index = int(probs.top1)
        classification_class = list(BOX_COLORS.keys())[classification_class_index]
        classification_class_conf = round(probs.top1conf.item(), 2)

        print(f"Detected class index: {classification_class_index}, Detected class: {classification_class}, Confidence: {classification_class_conf}")

        cls = classification_class if classification_class_conf > 0.9 else detection_class

        result_list.append({
            "bbox": [start_box, end_box],
            "class": cls,
            "confidence": classification_class_conf
        })

    # Print the results to the console
    for result in result_list:
        print(f"Bounding Box: {result['bbox']}, Class: {result['class']}, Confidence: {result['confidence']}")

    return result_list

if __name__ == "__main__":
    # Example usage
    image_path = "images/big-size.png"  # Replace with your image path
    detect(image_path)


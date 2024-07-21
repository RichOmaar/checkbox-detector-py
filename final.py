# Import libraries
import cv2  # for reading images, drawing bounding boxes
from ultralytics import YOLO 
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# Define constants
BOX_COLORS = {
    "unchecked": (0, 0, 255),  # red
    "checked": (0, 255, 0),    # green
    "block": (255, 255, 0)    # yellow
}
BOX_PADDING = 1

# Load models
DETECTION_MODEL = YOLO("models/detector-model.pt")
CLASSIFICATION_MODEL = YOLO("models/classifier-model.pt")  # 0: block, 1: checked, 2: unchecked

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def detect(image_path):
    """
    Output inference image with bounding box and text extracted using EasyOCR.
    Args:
    - image_path: Path to the image to check for checkboxes
    Return: image with bounding boxes and extracted text
    """
    image = cv2.imread(image_path)
    if image is None:
        return image

    # Predict on image
    results = DETECTION_MODEL.predict(source=image, conf=0.2, iou=0.8)  # Predict on image
    boxes = results[0].boxes  # Get bounding boxes

    if len(boxes) == 0:
        return image

    for box in boxes:
        # Get bounding box
        class_index = int(box.cls.item())
        confidence = box.conf.item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Class: {class_index}, Confidence: {confidence:.2f}")
        print(f"Bounding Box: Top-Left ({x1:.2f}, {y1:.2f}), Bottom-Right ({x2:.2f}, {y2:.2f})")

        # Crop the image to the bounding box
        roi = image[int(y1):int(y2), int(x1):int(x2)]

        # Use EasyOCR to extract text from the ROI
        result = reader.readtext(roi)
        text = " ".join([res[1] for res in result])
        print(f"Extracted Text: {text}")

        # Determine the color based on the class
        if class_index == 1:
            box_color = BOX_COLORS['checked']  # Green color for checked boxes
        else:
            box_color = BOX_COLORS['unchecked']  # Red color for unchecked boxes

        # Draw bounding box on the image
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), box_color, line_thickness)

        # Draw the extracted text on the image
        font_thickness = max(line_thickness - 1, 1)
        cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, font_thickness)

    # Save or display the image
    # output_path = "output_image.png"
    # cv2.imwrite(output_path, image)
    # print(f"Output image saved to {output_path}")

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return image

# Example usage
detect("images/ccl.jpg")

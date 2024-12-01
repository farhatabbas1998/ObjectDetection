import os
import cv2
from ultralytics import YOLO
import torch

def process_images(input_folder, output_folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO('yolov10n.pt').to(device)

    # List all files in the input folder
    input_files = os.listdir(input_folder)

    for file in input_files:
        image_path = os.path.join(input_folder, file)# Construct full file path and then load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image {file}. Skipping.")
            continue

        # Perform object detection
        results = model(image)

        # Extract detected objects, masks, and their bounding boxes
        marked_image = image.copy()

        # Loop through each detection
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            confidences = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Detected classes

            for box, conf, cls in zip(boxes, confidences, classes):
                # Draw the bounding box on the image
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

                # Write the class label and confidence score
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(
                    marked_image, 
                    label, 
                    (x1, max(0, y1 - 10)),  # Ensure the label is within image bounds
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    10, 
                    (0, 0, 255), 
                    10 
                )
        output_image_path = os.path.join(output_folder, file)
        cv2.imwrite(output_image_path, marked_image)
        print(f"Processed {file} and saved to {output_image_path}")

#----------------

input_folder = "Input_Folder"
output_folder = "Output_Folder"
process_images(input_folder, output_folder)
import os
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np
import time

def process_images(input_folder, output_folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(output_folder, exist_ok=True)
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device) #Mask R-CNN model with COCO weights
    model.eval()
    coco_classes = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.meta['categories']  # Get the COCO class labels
    transform = T.Compose([T.ToTensor()]) # Define a set of image transformations
 
    input_files = os.listdir(input_folder)

    for file in input_files:
        image_path = os.path.join(input_folder, file)# Load an image
        image = Image.open(image_path).convert("RGB")
        preprocess_start = time.time() # Start preprocessing timer
        input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        preprocess_end = time.time()

        # Perform object detection and instance segmentation
        inference_start = time.time()
        with torch.no_grad():
            prediction = model(input_image)
        inference_end = time.time()

        # Postprocess results
        labels = prediction[0]['labels'].cpu().numpy()
        masks = prediction[0]['masks'].cpu().numpy()
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        # Convert the image to OpenCV format for annotation
        marked_image = np.array(image)
        detected_objects = []

        for i in range(len(masks)):
            if scores[i] > 0.35:  # Threshold confidence
                mask = masks[i, 0]
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Draw bounding box
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(marked_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

                # Draw class label
                class_name = coco_classes[labels[i]]
                detected_objects.append(class_name)
                cv2.putText(marked_image, f'{class_name} {scores[i]:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
        output_image_path = os.path.join(output_folder, file)
        cv2.imwrite(output_image_path, cv2.cvtColor(marked_image, cv2.COLOR_RGB2BGR))
        postprocess_time = time.time() - inference_end # Calculate postprocess time
        print(f"{len(detected_objects)} objects detected: {', '.join(detected_objects)}")
        print(f"Processed {file} and saved to {output_image_path}")
        print(f"Speed: {preprocess_end - preprocess_start:.2f}ms preprocess, "
              f"{inference_end - inference_start:.2f}ms inference, "
              f"{postprocess_time:.2f}ms postprocess per image")

# Example usage:
input_folder = "Input_Folder"
output_folder = "Output_Folder"
process_images(input_folder, output_folder)

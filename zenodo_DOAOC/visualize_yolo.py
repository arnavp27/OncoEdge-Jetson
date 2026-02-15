import cv2
import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt

# CONFIGURATION
DATASET_DIR = r"E:\4thYear\EMbedding_AI\Project\datasets\oral_cancer_seg"
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

# CLASS NAMES (Must match your data.yaml)
CLASSES = {
    0: "OCA (Cancer) - RED",
    1: "OPMD (Pre-Cancer) - BLUE",
    2: "Benign - GREEN"
}
COLORS = {
    0: (255, 0, 0),    # Red
    1: (0, 0, 255),    # Blue
    2: (0, 255, 0)     # Green
}

def view_samples(num_samples=4):
    # Get all images
    img_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    if not img_paths:
        print("No images found! Check your path.")
        return

    # Pick random images
    random_samples = random.sample(img_paths, min(len(img_paths), num_samples))

    plt.figure(figsize=(15, 10))

    for i, img_path in enumerate(random_samples):
        # 1. Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to standard RGB
        h, w, _ = img.shape

        # 2. Load Label File
        label_name = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(LABELS_DIR, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 3. Draw Polygons
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:] # The rest are x1, y1, x2, y2...

                # Un-normalize coordinates (Math -> Pixels)
                points = []
                for j in range(0, len(coords), 2):
                    x = int(coords[j] * w)
                    y = int(coords[j+1] * h)
                    points.append([x, y])
                
                points = np.array(points, np.int32)
                points = points.reshape((-1, 1, 2))

                # Draw the shape
                color = COLORS.get(class_id, (255, 255, 255))
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=3)
                
                # Add Text Label
                label_text = CLASSES.get(class_id, "Unknown")
                cv2.putText(img, label_text, (points[0][0][0], points[0][0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Empty file = Healthy
            cv2.putText(img, "Healthy (No Label)", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show in grid
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(img_path))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_samples()
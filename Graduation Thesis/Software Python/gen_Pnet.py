import os
import cv2
import numpy as np
import glob
from pathlib import Path

# IoU function (replace with your utils.util.IoU if different)
def IoU(box, boxes):
    """
    Compute IoU between a box and multiple boxes.
    
    Args:
        box: numpy array [x1, y1, x2, y2]
        boxes: numpy array [N, 4] (x1, y1, x2, y2)
    
    Returns:
        iou: numpy array [N] of IoU values
    """
    # Box areas
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Intersection coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    # Intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Union area
    union = box_area + boxes_area - intersection
    iou = intersection / np.maximum(union, 1e-6)  # Avoid division by zero
    return iou

# Kaggle-specific paths
image_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/train/images"
labels_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/train/labels"
pos_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/positive_train"
part_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/part_train"
neg_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/negative_train"
anno_store_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store"

# Create directories
for directory in [pos_save_dir, part_save_dir, neg_save_dir, anno_store_dir]:
    os.makedirs(directory, exist_ok=True)

# Open annotation files
f1 = open(os.path.join(anno_store_dir, 'pos_12_train.txt'), 'w')
f2 = open(os.path.join(anno_store_dir, 'neg_12_train.txt'), 'w')
f3 = open(os.path.join(anno_store_dir, 'part_12_train.txt'), 'w')

# Get all label files
label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
print(f"{len(label_files)} label files found")

p_idx = 0  # Positive
n_idx = 0  # Negative
d_idx = 0  # Part (dont care)
idx = 0    # Processed images

for label_file in label_files:
    # Extract base filename and find corresponding image
    base_name = os.path.splitext(os.path.basename(label_file))[0]
    possible_extensions = ['.jpg', '.jpeg', '.png']
    image_name = None
    for ext in possible_extensions:
        candidate = f"{base_name}{ext}"
        if os.path.exists(os.path.join(image_dir, candidate)):
            image_name = candidate
            break
    
    if image_name is None:
        print(f"Warning: No matching image for {label_file}, skipping.")
        continue
    
    im_path = os.path.join(image_dir, image_name)
    img = cv2.imread(im_path)
    if img is None:
        print(f"Warning: Cannot read image {im_path}, skipping.")
        continue
    
    height, width, channel = img.shape
    
    # Read and convert YOLO annotations to x1, y1, x2, y2
    boxes = []
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    print(f"Warning: Invalid box format in {label_file}, line: {line}")
                    continue
                try:
                    x_center, y_center, w, h = map(float, parts[1:5])
                    # Convert YOLO (normalized) to x1, y1, x2, y2 (absolute)
                    x1 = (x_center - w / 2) * width
                    y1 = (y_center - h / 2) * height
                    x2 = (x_center + w / 2) * width
                    y2 = (y_center + h / 2) * height
                    if x2 > x1 and y2 > y1 and 0 <= x1 < width and 0 <= y2 < height:
                        boxes.append([x1, y1, x2, y2])
                    else:
                        print(f"Warning: Invalid box dimensions in {label_file}, line: {line}")
                except (ValueError, IndexError):
                    print(f"Warning: Invalid box format in {label_file}, line: {line}")
                    continue
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        continue
    
    boxes = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
    idx += 1

    # Generate negative samples
    neg_num = 0
    while neg_num < 35:
        size = np.random.randint(12, min(width, height) / 2)
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        if len(boxes) == 0 or np.max(IoU(crop_box, boxes)) < 0.3:
            cropped_im = img[ny:ny + size, nx:nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            save_file = os.path.join(neg_save_dir, f"{n_idx}.jpg").replace("\\", "/")
            f2.write(f"{save_file} 0\n")
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    # Process each ground truth box
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        # Ignore small or invalid faces
        if max(w, h) < 40 or x1 < 0 or y1 < 0 or w <= 0 or h <= 0:
            continue

        # Generate additional negative samples near the face
        for i in range(5):
            size = np.random.randint(12, min(width, height) / 2)
            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)

            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            if np.max(IoU(crop_box, boxes)) < 0.3:
                cropped_im = img[int(ny1):int(ny1 + size), int(nx1):int(nx1 + size), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(neg_save_dir, f"{n_idx}.jpg").replace("\\", "/")
                f2.write(f"{save_file} 0\n")
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # Generate positive and part samples
        for i in range(20):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / size
            offset_y1 = (y1 - ny1) / size
            offset_x2 = (x2 - nx2) / size
            offset_y2 = (y2 - ny2) / size

            cropped_im = img[int(ny1):int(ny2), int(nx1):int(nx2), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            iou = IoU(crop_box, box_)
            if iou >= 0.65:
                save_file = os.path.join(pos_save_dir, f"{p_idx}.jpg").replace("\\", "/")
                f1.write(f"{save_file} 1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n")
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif iou >= 0.4 and d_idx < 1.2 * p_idx + 1:
                save_file = os.path.join(part_save_dir, f"{d_idx}.jpg").replace("\\", "/")
                f3.write(f"{save_file} -1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n")
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

    print(f"{idx} images done, pos: {p_idx}, part: {d_idx}, neg: {n_idx}")

f1.close()
f2.close()
f3.close()
import sys
import cv2
import os
import numpy as np
import torch
from utils.util import IoU
from MTCNN import create_mtcnn_net

# Directory and file paths
prefix = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/valid/images"
anno_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_valid_processed.txt"
pos_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/positive_val_48"
part_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/part_val_48"
neg_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/negative_val_48"
anno_store_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store"

# Create directories if they don't exist
os.makedirs(pos_save_dir, exist_ok=True)
os.makedirs(part_save_dir, exist_ok=True)
os.makedirs(neg_save_dir, exist_ok=True)
os.makedirs(anno_store_dir, exist_ok=True)

# Open files to store labels
f1 = open(os.path.join(anno_store_dir, 'pos_48_val.txt'), 'w')
f2 = open(os.path.join(anno_store_dir, 'neg_48_val.txt'), 'w')
f3 = open(os.path.join(anno_store_dir, 'part_48_val.txt'), 'w')

# Read annotation file
try:
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print(f"{num} pics in total")
except FileNotFoundError:
    print(f"Error: Annotation file {anno_file} not found.")
    sys.exit(1)

image_size = 48
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

p_idx = 0  # Positive image counter
n_idx = 0  # Negative image counter
d_idx = 0  # Part image counter
idx = 0    # Processed image counter

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = os.path.join(prefix, annotation[0])
    print(f"Processing: {im_path}")

    # Load and validate image
    image = cv2.imread(im_path)
    if image is None:
        print(f"Error: Could not load image {im_path}. Skipping...")
        continue

    # Parse bounding box annotations
    try:
        bbox = list(map(float, annotation[1:]))
        # Check if there are any bounding box coordinates
        if len(bbox) < 4:  # Need at least one box (x1, y1, w, h)
            print(f"Error: No bounding boxes found for {im_path}. Skipping...")
            continue
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        # Convert from x1, y1, w, h to x1, y1, x2, y2
        boxes[:, 2] += boxes[:, 0] - 1
        boxes[:, 3] += boxes[:, 1] - 1
    except ValueError as e:
        print(f"Error: Invalid bounding box format in annotation for {im_path}: {e}. Skipping...")
        continue

    # Run MTCNN detection (PNet and RNet)
    try:
        bboxes, landmarks = create_mtcnn_net(
            image, 12, device,
            p_model_path="C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Pnet_weight_v2.pth",
            r_model_path="C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Rnet_weight_v2.pth"
        )
        dets = np.round(bboxes[:, 0:4]) if bboxes is not None else np.array([])
    except Exception as e:
        print(f"Error: MTCNN detection failed for {im_path}: {e}. Skipping...")
        continue

    if dets.shape[0] == 0:
        print(f"No detections found for {im_path}. Skipping...")
        continue

    idx += 1
    height, width, channel = image.shape

    for box in dets:
        x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
        box_width = x_right - x_left + 1
        box_height = y_bottom - y_top + 1

        # Ignore boxes that are too small or beyond image borders
        if (box_width < 20 or box_height < 20 or
                x_left < 0 or y_top < 0 or
                x_right > width - 1 or y_bottom > height - 1):
            continue

       

        # Compute IoU between detected box and ground truth boxes
        try:
            iou = IoU(box, boxes)
            if len(iou) == 0:
                print(f"Warning: IoU returned empty array for {im_path}, box: {box}. Skipping box...")
                continue
        except Exception as e:
            print(f"Error: IoU computation failed for {im_path}, box: {box}: {e}. Skipping box...")
            continue

        cropped_im = image[y_top:y_bottom + 1, x_left:x_right + 1, :]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # Save negative images (IoU < 0.3)
        if np.max(iou) < 0.3 and n_idx < 3.2 * p_idx + 1:
            save_file = os.path.join(neg_save_dir, f"{n_idx}.jpg").replace("\\", "/")
            f2.write(f"{save_file} 0\n")
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
        else:
            # Find ground truth box with highest IoU
            idx_iou = np.argmax(iou)
            assigned_gt = boxes[idx_iou]
            x1, y1, x2, y2 = assigned_gt

            # Compute bounding box regression offsets
            offset_x1 = (x1 - x_left) / float(box_width)
            offset_y1 = (y1 - y_top) / float(box_height)
            offset_x2 = (x2 - x_right) / float(box_width)
            offset_y2 = (y2 - y_bottom) / float(box_height)

            # Save positive images (IoU >= 0.65)
            if np.max(iou) >= 0.4:
                save_file = os.path.join(pos_save_dir, f"{p_idx}.jpg").replace("\\", "/")
                f1.write(f"{save_file} 1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n")
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            # Save part images (0.4 <= IoU < 0.65)
            elif np.max(iou) >= 0.3 and d_idx < 1.2 * p_idx + 1:
                save_file = os.path.join(part_save_dir, f"{d_idx}.jpg").replace("\\", "/")
                f3.write(f"{save_file} -1 {offset_x1:.2f} {offset_y1:.2f} {offset_x2:.2f} {offset_y2:.2f}\n")
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

    print(f"{idx} images done, pos: {p_idx}, part: {d_idx}, neg: {n_idx}")

# Close label files
f1.close()
f2.close()
f3.close()
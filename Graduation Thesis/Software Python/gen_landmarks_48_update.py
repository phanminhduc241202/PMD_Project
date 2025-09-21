# coding: utf-8
import os
import cv2
import random
import numpy as np
import glob

# Assuming IoU is defined in utils.util
from utils.util import IoU

# Dataset paths from data.yaml
data_base_dir = ""  # Adjust if your dataset has a base directory
train_img_dir = os.path.join(data_base_dir, "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/landmarks/train/images").replace("\\", "/")
train_label_dir = os.path.join(data_base_dir, "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/landmarks/train/labels").replace("\\", "/")
val_img_dir = os.path.join(data_base_dir, "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/landmarks/valid/images").replace("\\", "/")
val_label_dir = os.path.join(data_base_dir, "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/landmarks/valid/labels").replace("\\", "/")

size = 48  # Target size for O-Net

def process_dataset(img_dir, label_dir, output_dir, anno_filename):
    # Output directories
    landmark_imgs_save_dir = os.path.join(output_dir, "landmark").replace("\\", "/")
    if not os.path.exists(landmark_imgs_save_dir):
        os.makedirs(landmark_imgs_save_dir)

    anno_dir = os.path.join(output_dir, "anno_store").replace("\\", "/")
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    save_landmark_anno = os.path.join(anno_dir, anno_filename).replace("\\", "/")

    # Open the output annotation file
    f = open(save_landmark_anno, 'w')

    # Collect image files
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    img_files = [f.replace("\\", "/") for f in img_files]
    num = len(img_files)
    print(f"Processing {num} images from {img_dir}")

    l_idx = 0
    idx = 0

    for img_path in img_files:
        # Determine corresponding label file
        label_path = img_path.replace(img_dir, label_dir).rsplit(".", 1)[0] + ".txt"

        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {img_path}")
            continue

        # Load image
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image: {img_path}")
                continue
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        height, width, channel = img.shape

        # Read label file
        try:
            with open(label_path, 'r') as f2:
                lines = f2.readlines()
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
            continue

        if not lines:
            print(f"Warning: Empty label file {label_path}")
            continue

        # Process the first annotation (assuming one face per image)
        annotation = lines[0].strip().split()
        # Expected format: class_id center_x center_y width height x1 y1 v1 x2 y2 v2 ... x5 y5 v5 (normalized)
        if len(annotation) < 1 + 4 + 5 * 3:
            print(f"Warning: Invalid label format in {label_path}: {annotation}")
            continue

        # Extract landmarks (x, y only, ignoring visibility)
        landmark = []
        for i in range(5):
            x = float(annotation[5 + i * 3]) * width
            y = float(annotation[5 + i * 3 + 1]) * height
            landmark.extend([x, y])
        landmark = np.array(landmark, dtype=np.float32)

        # Derive gt_box from landmarks
        x_coords = landmark[0::2]
        y_coords = landmark[1::2]
        x1 = max(int(np.min(x_coords) - 20), 0)
        y1 = max(int(np.min(y_coords) - 20), 0)
        x2 = min(int(np.max(x_coords) + 20), width - 1)
        y2 = min(int(np.max(y_coords) + 20), height - 1)
        gt_box = np.array([x1, y1, x2, y2], dtype=np.int32)

        idx += 1
        if idx % 100 == 0:
            print("%d images done, landmark images: %d" % (idx, l_idx))

        # gt's width and height
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # Random shift for data augmentation
        for i in range(15):
            low = int(min(w, h) * 0.8)
            high = int(np.ceil(1.25 * max(w, h)))
            if low >= high:
                low = high - 1
            if low < 1:
                low = 1
            if high < 2:
                high = 2
            bbox_size = np.random.randint(low, high)
            if bbox_size <= 0:
                print(f"Warning: Invalid bbox_size={bbox_size} for image {img_path}, w={w}, h={h}")
                continue

            delta_x_low = max(int(-w * 0.2), -bbox_size // 2)
            delta_x_high = min(int(w * 0.2) + 1, bbox_size // 2)
            delta_y_low = max(int(-h * 0.2), -bbox_size // 2)
            delta_y_high = min(int(h * 0.2) + 1, bbox_size // 2)

            if delta_x_low >= delta_x_high:
                delta_x = 0
            else:
                delta_x = np.random.randint(delta_x_low, delta_x_high)

            if delta_y_low >= delta_y_high:
                delta_y = 0
            else:
                delta_y = np.random.randint(delta_y_low, delta_y_high)

            nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)
            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size

            if nx2 <= nx1 or ny2 <= ny1:
                print(f"Warning: Invalid crop region for image {img_path}: nx1={nx1}, nx2={nx2}, ny1={ny1}, ny2={ny2}, bbox_size={bbox_size}")
                continue

            if nx2 > width or ny2 > height:
                continue

            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
            if cropped_im.size == 0:
                print(f"Warning: Empty cropped image for {img_path}: nx1={nx1}, nx2={nx2}, ny1={ny1}, ny2={ny2}")
                continue

            resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)

            # Calculate offsets for bounding box
            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            # Calculate normalized landmark offsets (5 keypoints)
            offset_landmarks = []
            for i in range(0, len(landmark), 2):
                offset_x = (landmark[i] - nx1) / float(bbox_size)
                offset_y = (landmark[i + 1] - ny1) / float(bbox_size)
                offset_landmarks.extend([offset_x, offset_y])

            # Calculate IoU
            iou = IoU(crop_box.astype(np.float32), np.expand_dims(gt_box.astype(np.float32), 0))
            if iou > 0.65:
                save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx).replace("\\", "/")
                cv2.imwrite(save_file, resized_im)

                # Write to annotation file (adjusted for 5 landmarks: 4 box offsets + 10 landmark offsets)
                f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2,
                    offset_landmarks[0], offset_landmarks[2], offset_landmarks[4], offset_landmarks[6], offset_landmarks[8],
                    offset_landmarks[1], offset_landmarks[3], offset_landmarks[5], offset_landmarks[7], offset_landmarks[9]))
                l_idx += 1

    f.close()
    print(f"Processed {idx} images, generated {l_idx} landmark images for {img_dir}.")

# Process train dataset
train_output_dir = "./output/landmarks_48"
process_dataset(train_img_dir, train_label_dir, train_output_dir, "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/landmark_48_train.txt")

# Process valid dataset
valid_output_dir = "./output/landmarks_48"
process_dataset(val_img_dir, val_label_dir, valid_output_dir, "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/landmark_48_valid.txt")
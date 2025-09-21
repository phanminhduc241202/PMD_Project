# coding: utf-8
import os
import cv2
import random
import sys
import numpy as np
import time
from utils.util import IoU  # Assuming IoU is defined in utils.util

# Paths (adjust these according to your dataset's structure)
prefix = ''
data_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/img_align_celeba"  # Directory where images are stored
anno_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/list_landmarks_celeba.txt"  # Path to your dataset file

size = 48  # Target size for O-Net
image_id = 0

# Output directories
landmark_imgs_save_dir = os.path.join(data_dir, "output/landmark").replace("\\", "/")
if not os.path.exists(landmark_imgs_save_dir):
    os.makedirs(landmark_imgs_save_dir)

anno_dir = './output/anno_store'
if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)

landmark_anno_filename = "landmark_48.txt"
save_landmark_anno = os.path.join(anno_dir, landmark_anno_filename).replace("\\", "/")

# Open the output annotation file
f = open(save_landmark_anno, 'w')

# Read the dataset
with open(anno_file, 'r') as f2:
    annotations = f2.readlines()

# Skip the header lines
annotations = annotations[2:]  # Skip "202599" and the column names
num = len(annotations)
print("%d total images" % num)

l_idx = 0
idx = 0

for annotation in annotations:
    annotation = annotation.strip().split()
    
    # Expecting 11 elements: image_name, lefteye_x, lefteye_y, ..., rightmouth_y
    assert len(annotation) == 11, f"Each line should have 11 elements, got {len(annotation)}: {annotation}"

    # Image path
    image_name = annotation[0]
    im_path = os.path.join(data_dir, image_name)

    # Load image with error handling
    try:
        img = cv2.imread(im_path)
        if img is None:
            print(f"Warning: Failed to load image: {im_path}")
            continue
    except Exception as e:
        print(f"Error loading image {im_path}: {e}")
        continue

    height, width, channel = img.shape

    # Extract landmark coordinates
    landmark = list(map(float, annotation[1:]))
    landmark = np.array(landmark, dtype=np.float32)
    # landmark order: [lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y]

    # Derive gt_box from landmarks
    x_coords = landmark[0::2]
    y_coords = landmark[1::2]
    x1 = max(int(np.min(x_coords) - 20), 0)
    y1 = max(int(np.min(y_coords) - 20), 0)
    x2 = min(int(np.max(x_coords) + 20), width - 1)
    y2 = min(int(np.max(y_coords) + 20), height - 1)
    gt_box = np.array([x1, y1, x2, y2], dtype=np.int32)
    x1, y1, x2, y2 = gt_box

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
        # Ensure bbox_size is at least 1
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
            print(f"Warning: Invalid bbox_size={bbox_size} for image {image_name}, w={w}, h={h}")
            continue

        # Ensure the range for delta_x and delta_y is valid
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

        # Check if the cropping region is valid
        if nx2 <= nx1 or ny2 <= ny1:
            print(f"Warning: Invalid crop region for image {image_name}: nx1={nx1}, nx2={nx2}, ny1={ny1}, ny2={ny2}, bbox_size={bbox_size}")
            continue

        if nx2 > width or ny2 > height:
            continue

        crop_box = np.array([nx1, ny1, nx2, ny2])
        cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
        if cropped_im.size == 0:
            print(f"Warning: Empty cropped image for {image_name}: nx1={nx1}, nx2={nx2}, ny1={ny1}, ny2={ny2}")
            continue

        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)

        # Calculate offsets for bounding box
        offset_x1 = (x1 - nx1) / float(bbox_size)
        offset_y1 = (y1 - ny1) / float(bbox_size)
        offset_x2 = (x2 - nx2) / float(bbox_size)
        offset_y2 = (y2 - ny2) / float(bbox_size)

        # Calculate normalized landmark offsets
        offset_left_eye_x = (landmark[0] - nx1) / float(bbox_size)
        offset_left_eye_y = (landmark[1] - ny1) / float(bbox_size)
        offset_right_eye_x = (landmark[2] - nx1) / float(bbox_size)
        offset_right_eye_y = (landmark[3] - ny1) / float(bbox_size)
        offset_nose_x = (landmark[4] - nx1) / float(bbox_size)
        offset_nose_y = (landmark[5] - ny1) / float(bbox_size)
        offset_left_mouth_x = (landmark[6] - nx1) / float(bbox_size)
        offset_left_mouth_y = (landmark[7] - ny1) / float(bbox_size)
        offset_right_mouth_x = (landmark[8] - nx1) / float(bbox_size)
        offset_right_mouth_y = (landmark[9] - ny1) / float(bbox_size)

        # Calculate IoU
        iou = IoU(crop_box.astype(np.float32), np.expand_dims(gt_box.astype(np.float32), 0))
        if iou > 0.65:
            save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx).replace("\\", "/")
            cv2.imwrite(save_file, resized_im)

            # Write to annotation file
            f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % (
                offset_x1, offset_y1, offset_x2, offset_y2,
                offset_left_eye_x, offset_right_eye_x, offset_nose_x, offset_left_mouth_x, offset_right_mouth_x,
                offset_left_eye_y, offset_right_eye_y, offset_nose_y, offset_left_mouth_y, offset_right_mouth_y))
            l_idx += 1

f.close()
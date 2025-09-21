import sys
sys.path.append('..')
import cv2
import os
import numpy as np
from utils.util import IoU, preprocess
import torch
from MTCNN import create_mtcnn_net

# Đường dẫn từ file gốc
prefix = ''
anno_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_valid_processed.txt"
im_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/valid/images"
pos_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/positive_val_24"
part_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/part_val_24"
neg_save_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/negative_val_24"
anno_store_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store"

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir, exist_ok=True)
if not os.path.exists(part_save_dir):
    os.makedirs(part_save_dir, exist_ok=True)
if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir, exist_ok=True)
if not os.path.exists(anno_store_dir):
    os.makedirs(anno_store_dir, exist_ok=True)

# Mở file annotation
f1 = open(os.path.join(anno_store_dir, 'pos_24_val.txt'), 'w')
f2 = open(os.path.join(anno_store_dir, 'neg_24_val.txt'), 'w')
f3 = open(os.path.join(anno_store_dir, 'part_24_val.txt'), 'w')

# Đọc file annotation
try:
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)
except FileNotFoundError:
    print(f"Error: Annotation file {anno_file} not found.")
    sys.exit(1)

image_size = 24
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # dont care
idx = 0    # processed images
skipped = 0  # số lượng ảnh bị bỏ qua

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    if len(annotation) < 5:  # Kiểm tra định dạng annotation
        print(f"Warning: Invalid annotation format: {annotation}")
        skipped += 1
        continue
    
    im_path = os.path.join(im_dir, annotation[0])
    print(f"Processing {im_path}")
    
    # Chuyển đổi bounding box từ [x1, y1, w, h] sang [x1, y1, x2, y2]
    try:
        # Kiểm tra tất cả giá trị có thể chuyển thành float
        bbox = []
        for val in annotation[1:]:
            try:
                float_val = float(val)
                bbox.append(float_val)
            except ValueError:
                print(f"Error: Invalid value '{val}' in bounding box for {im_path}")
                raise ValueError("Invalid bounding box value")
        
        # Đảm bảo số lượng giá trị đủ để tạo ít nhất một box (4 giá trị: x1, y1, w, h)
        if len(bbox) < 4:
            print(f"Error: Insufficient bounding box values in {im_path}: {bbox}")
            skipped += 1
            continue
        
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2] += boxes[:, 0] - 1  # x2 = x1 + w - 1
        boxes[:, 3] += boxes[:, 1] - 1  # y2 = y1 + h - 1
    except (ValueError, IndexError) as e:
        print(f"Error parsing bounding box in {im_path}: {e}")
        skipped += 1
        continue

    # Đọc ảnh
    image = cv2.imread(im_path)
    if image is None:
        print(f"Warning: Cannot read image {im_path}, skipping.")
        skipped += 1
        continue

    # Chạy PNet để lấy đề xuất
    try:
        bboxes, landmarks = create_mtcnn_net(image, 12, device, p_model_path='C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Pnet_weight_v2.pth')
        dets = np.round(bboxes[:, 0:4]) if bboxes is not None and len(bboxes) > 0 else np.array([])
        print(f"PNet proposals: {len(dets)}")
    except Exception as e:
        print(f"Error running PNet on {im_path}: {e}")
        skipped += 1
        continue

    if dets.shape[0] == 0:
        print(f"No proposals from PNet for {im_path}, skipping.")
        continue

    idx += 1
    height, width, channel = image.shape

    for box in dets:
        x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
        width = x_right - x_left + 1
        height = y_bottom - y_top + 1

        # Bỏ qua box nhỏ hoặc ngoài biên ảnh
        if width < 20 or height < 20 or x_left < 0 or y_top < 0 or x_right > image.shape[1] - 1 or y_bottom > image.shape[0] - 1:
            continue

        # Tính IoU với ground truth boxes
        iou = IoU(box, boxes)
        cropped_im = image[y_top:y_bottom + 1, x_left:x_right + 1, :]
        try:
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error resizing box {box} in {im_path}: {e}")
            continue

        # Lưu negative samples
        if np.max(iou) < 0.3 and n_idx < 3.2 * p_idx + 1:
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx).replace("\\", "/")
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
        else:
            # Tìm ground truth box có IoU cao nhất
            idx_iou = np.argmax(iou)
            assigned_gt = boxes[idx_iou]
            x1, y1, x2, y2 = assigned_gt

            # Tính offset
            offset_x1 = (x1 - x_left) / float(width)
            offset_y1 = (y1 - y_top) / float(height)
            offset_x2 = (x2 - x_right) / float(width)
            offset_y2 = (y2 - y_bottom) / float(height)

            # Lưu positive samples
            if np.max(iou) >= 0.4:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx).replace("\\", "/")
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

            # Lưu part samples
            elif np.max(iou) >= 0.3 and d_idx < 1.2 * p_idx + 1:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx).replace("\\", "/")
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

    print("%s images done, pos: %s part: %s neg: %s, skipped: %s" % (idx, p_idx, d_idx, n_idx, skipped))

f1.close()
f2.close()
f3.close()
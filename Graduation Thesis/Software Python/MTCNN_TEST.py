import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from MTCNN_nets import PNet, RNet, ONet
import math

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load models and weights
pnet = PNet().to(device)
rnet = RNet().to(device)
onet = ONet().to(device)

pnet.load_state_dict(torch.load('C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Pnet_weight.pth', map_location=device))
rnet.load_state_dict(torch.load('C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Rnet_weight.pth', map_location=device))
onet.load_state_dict(torch.load('C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Onet_weight.pth', map_location=device))

pnet.eval()
rnet.eval()
onet.eval()

# Image loading
image_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/img/32.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image {image_path}")
    exit()
height, width, channel = image.shape
min_length = min(height, width)
print(f"Image shape: {image.shape}")

# Parameters
min_face_size = 30.0
thresholds = [0.6, 0.7, 0.6]  # Reduced O-Net threshold to 0.6
nms_thresholds = [0.7, 0.7, 0.5]
min_detection_size = 12
factor = 0.707

# Preprocessing function
def preprocess(img):
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.asarray(img, 'float32')
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img

# NMS function
def nms(boxes, overlap_threshold=0.5, mode='union'):
    if len(boxes) == 0:
        return []
    pick = []
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)
    while len(ids) > 0:
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)
        inter = w * h
        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        else:
            overlap = inter / (area[i] + area[ids[:last]] - inter)
        ids = np.delete(ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]]))
    return pick

# Calibrate bounding boxes
def calibrate_box(bboxes, offsets):
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)
    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

# Convert to square boxes with scale factor
def convert_to_square(bboxes, scale_factor=0.9):
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w) * scale_factor
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    square_bboxes[:, 4] = bboxes[:, 4]
    return square_bboxes

# Correct bounding boxes
def correct_bboxes(bboxes, width, height):
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]
    x, y, ex, ey = x1, y1, x2, y2
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0
    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]
    return return_list

# Stage 1: P-Net
scales = []
m = min_detection_size / min_face_size
min_length *= m
factor_count = 0
while min_length > min_detection_size:
    scales.append(m * factor ** factor_count)
    min_length *= factor
    factor_count += 1

bounding_boxes = []
with torch.no_grad():
    for scale in scales:
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
        img = torch.FloatTensor(preprocess(img)).to(device)
        offset, prob = pnet(img)
        probs = prob.cpu().data.numpy()[0, 1, :, :]
        offsets = offset.cpu().data.numpy()
        stride, cell_size = 2, 12
        inds = np.where(probs > thresholds[0])
        if inds[0].size == 0:
            boxes = None
        else:
            tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
            offsets = np.array([tx1, ty1, tx2, ty2])
            score = probs[inds[0], inds[1]]
            bounding_box = np.vstack([
                np.round((stride * inds[1] + 1.0) / scale),
                np.round((stride * inds[0] + 1.0) / scale),
                np.round((stride * inds[1] + 1.0 + cell_size) / scale),
                np.round((stride * inds[0] + 1.0 + cell_size) / scale),
                score, offsets
            ])
            boxes = bounding_box.T
            keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
            boxes = boxes[keep]
        if boxes is not None:
            bounding_boxes.append(boxes)

# Combine and process P-Net boxes
bounding_boxes = [i for i in bounding_boxes if i is not None]
if len(bounding_boxes) == 0:
    print("No boxes detected by P-Net")
    exit()
bounding_boxes = np.vstack(bounding_boxes)
print(f"P-Net: {bounding_boxes.shape[0]} boxes before NMS")
keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
bounding_boxes = bounding_boxes[keep]
bboxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
bboxes = convert_to_square(bboxes, scale_factor=0.9)
bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
print(f"P-Net: {bboxes.shape[0]} boxes after processing")

# Visualize P-Net results
pnet_img = image.copy()
for i in range(bboxes.shape[0]):
    bbox = bboxes[i, :4]
    cv2.rectangle(pnet_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

# Stage 2: R-Net
size = 24
num_boxes = len(bboxes)
print(f"R-Net: Received {num_boxes} boxes from P-Net")
[dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)
img_boxes = np.zeros((num_boxes, 3, size, size))
for i in range(num_boxes):
    img_box = np.zeros((h[i], w[i], 3))
    try:
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
    except ValueError as e:
        print(f"Warning: Size mismatch for R-Net box {i}: {bboxes[i, :4]}, img_box shape: {img_box.shape}, image slice shape: {image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :].shape}")
        continue
 
    img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)
    img_boxes[i, :, :, :] = preprocess(img_box)

if num_boxes > 0:
    img_boxes = torch.FloatTensor(img_boxes).to(device)
    offset, prob = rnet(img_boxes)
    offsets = offset.cpu().data.numpy()
    probs = prob.cpu().data.numpy()
    print(f"R-Net: Probabilities shape: {probs.shape}, max score: {np.max(probs[:, 1]) if probs.size > 0 else 'N/A'}")
    keep = np.where(probs[:, 1] > thresholds[1])[0]
    print(f"R-Net: {len(keep)} boxes above threshold {thresholds[1]}")
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    keep = nms(bboxes, nms_thresholds[1])
    bboxes = bboxes[keep]
    bboxes = calibrate_box(bboxes, offsets[keep])
    bboxes = convert_to_square(bboxes, scale_factor=0.9)
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
    print(f"R-Net: {bboxes.shape[0]} boxes after processing")
else:
    bboxes = np.array([])
    print("R-Net: No boxes to process")

# Visualize R-Net results
rnet_img = image.copy()
for i in range(bboxes.shape[0]):
    bbox = bboxes[i, :4]
    cv2.rectangle(rnet_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)

# Stage 3: O-Net (No Landmarks)
size = 48
num_boxes = len(bboxes)
print(f"O-Net: Received {num_boxes} boxes from R-Net")
if num_boxes > 0:
    # Shrink bounding boxes before O-Net
    shrink_factor = 0.95  # Increased to 0.95
    bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) * (1 - shrink_factor) / 2
    bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) * (1 - shrink_factor) / 2
    bboxes[:, 2] = bboxes[:, 2] - (bboxes[:, 2] - bboxes[:, 0]) * (1 - shrink_factor) / 2
    bboxes[:, 3] = bboxes[:, 3] - (bboxes[:, 3] - bboxes[:, 1]) * (1 - shrink_factor) / 2

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size))
    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))
        try:
            img_region = image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
            if img_region.shape[0] != h[i] or img_region.shape[1] != w[i]:
                print(f"Warning: Adjusting region for O-Net box {i}: expected ({h[i]}, {w[i]}), got {img_region.shape[:2]}")
                img_region = cv2.resize(img_region, (w[i], h[i]), interpolation=cv2.INTER_LINEAR)
            img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = img_region
        except ValueError as e:
            print(f"Warning: Size mismatch for O-Net box {i}: {bboxes[i, :4]}, img_box shape: {img_box.shape}, image slice shape: {img_region.shape}")
            continue
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)
        img_boxes[i, :, :, :] = preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    landmarks, offset, prob = onet(img_boxes)
    offsets = offset.cpu().data.numpy()
    probs = prob.cpu().data.numpy()
    print(f"O-Net: Probabilities shape: {probs.shape}, max score: {np.max(probs[:, 1]) if probs.size > 0 else 'N/A'}")
    # Debug: Print probabilities for each box
    for i, score in enumerate(probs[:, 1]):
        print(f"O-Net box {i}: score = {score}")
    keep = np.where(probs[:, 1] > thresholds[2])[0]
    print(f"O-Net: {len(keep)} boxes above threshold {thresholds[2]}")
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    keep = nms(bboxes, nms_thresholds[2], mode='min')
    bboxes = bboxes[keep]
    bboxes = calibrate_box(bboxes, offsets[keep])
    bboxes = convert_to_square(bboxes, scale_factor=0.9)
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
    print(f"O-Net: {bboxes.shape[0]} boxes after processing")
else:
    bboxes = np.array([])
    print("O-Net: No boxes to process")

# Visualize O-Net results
onet_img = image.copy()
for i in range(bboxes.shape[0]):
    bbox = bboxes[i, :4]
    cv2.rectangle(onet_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

# Display all stages
plt.figure(figsize=(15, 5))

# P-Net
plt.subplot(1, 3, 1)
plt.imshow(pnet_img[:, :, ::-1])
plt.title('P-Net: Candidate Boxes')
plt.axis('off')

# R-Net
plt.subplot(1, 3, 2)
plt.imshow(rnet_img[:, :, ::-1])
plt.title('R-Net: Refined Boxes')
plt.axis('off')

# O-Net
plt.subplot(1, 3, 3)
plt.imshow(onet_img[:, :, ::-1])
plt.title('O-Net: Final Boxes')
plt.axis('off')

plt.tight_layout()
plt.savefig('face_detection_results.png')
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from collections import OrderedDict

# Định nghĩa lớp PNet
class PNet(nn.Module):
    def __init__(self, is_train=False):
        super(PNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),
            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),
            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)

        if self.is_train is False:
            a = F.softmax(a, dim=1)

        return b, a

# Hàm NMS
def nms(boxes, scores, threshold=0.5):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

# Hàm tạo image pyramid
def generate_image_pyramid(img, min_size=12, factor=0.709):
    height, width = img.shape[:2]
    scales = []
    s = min_size / min(height, width)
    while height * s >= min_size and width * s >= min_size:
        scales.append(s)
        s *= factor
    print(f"Generated scales: {scales}")
    return scales

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {img_rgb.shape}")
    return img_rgb

# Hàm chuyển đổi offset thành tọa độ bounding box
def convert_to_boxes(cls, bbox, scale, stride=2, min_size=12, threshold=0.1):
    cls = cls[:, 1, :, :]  # Lấy xác suất face
    height, width = cls.shape[1:]
    offset = bbox.permute(0, 2, 3, 1).cpu().numpy()[0]  # [h, w, 4]
    cls = cls.cpu().numpy()[0]  # [h, w]

    boxes = []
    scores = []
    for i in range(height):
        for j in range(width):
            if cls[i, j] > threshold:
                dx, dy, dw, dh = offset[i, j]
                x1 = (j * stride) / scale
                y1 = (i * stride) / scale
                x2 = ((j * stride) + min_size) / scale
                y2 = ((i * stride) + min_size) / scale

                # Điều chỉnh bounding box theo offset
                center_x = x1 + (x2 - x1) * 0.5
                center_y = y1 + (y2 - y1) * 0.5
                width = (x2 - x1) * np.exp(dw)
                height = (y2 - y1) * np.exp(dh)

                x1 = center_x - width * 0.5 + dx * (x2 - x1)
                y1 = center_y - height * 0.5 + dy * (y2 - y1)
                x2 = center_x + width * 0.5 + dx * (x2 - x1)
                y2 = center_y + height * 0.5 + dy * (y2 - y1)

                boxes.append([x1, y1, x2, y2])
                scores.append(cls[i, j])

    print(f"Scale {scale}: Found {len(boxes)} boxes with threshold {threshold}")
    return np.array(boxes), np.array(scores)

# Hàm chạy PNet trên image pyramid
def run_pnet_on_pyramid(model, img_rgb, scales, device='cpu', threshold=0.001):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    all_boxes = []
    all_scores = []
    
    for scale in scales:
        scaled_img = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(scaled_img)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            bbox, cls = model(img_tensor)
        
        boxes, scores = convert_to_boxes(cls, bbox, scale, threshold=threshold)
        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_scores.append(scores)
    
    if len(all_boxes) == 0:
        print("No boxes detected across all scales.")
        return np.array([]), np.array([])
    
    all_boxes = np.vstack(all_boxes)
    all_scores = np.hstack(all_scores)
    
    keep = nms(all_boxes, all_scores, threshold=0.2)
    print(f"After NMS: {len(keep)} boxes remaining.")
    return all_boxes[keep], all_scores[keep]

# Hàm vẽ bounding box
def draw_boxes(img, boxes, scores):
    img_copy = img.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img_copy

# Hàm load mô hình PNet
def load_pnet_model(model_path, device='cpu'):
    model = PNet(is_train=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Main
if __name__ == "__main__":
    # Thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Đường dẫn
    model_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Pnet_weight.pth"  # Đảm bảo file này tồn tại
    image_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/img/2.jpg"  # Thay bằng đường dẫn ảnh của bạn
    
    # Load mô hình
    pnet = load_pnet_model(model_path, device)
    
    # Tiền xử lý ảnh
    img_rgb = preprocess_image(image_path)
    
    # Tạo image pyramid
    scales = generate_image_pyramid(img_rgb)
    
    # Chạy PNet trên pyramid với ngưỡng thấp hơn
    boxes, scores = run_pnet_on_pyramid(pnet, img_rgb, scales, device, threshold=0.0001)
    
    # In kết quả
    print("Detected boxes:", boxes)
    print("Scores:", scores)
    
    # Vẽ bounding box
    img_with_boxes = draw_boxes(img_rgb, boxes, scores)
    
    # Lưu ảnh kết quả
    cv2.imwrite('output_image.jpg', cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print("Saved output image as 'output_image.jpg'")
    
    # Hiển thị ảnh
    cv2.imshow('Image with Bounding Boxes', cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

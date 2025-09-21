# import os
# import math
# import cv2
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from Quantization_MTCNNv2 import QuantPNet, QuantRNet, QuantONet

# # Workaround for multiple OpenMP runtimes on Windows
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # ---- Config paths ----
# WEIGHT_DIR = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights"
# weight_paths = {
#     'pnet': os.path.join(WEIGHT_DIR, "Quan_Pnetv2_weight.pth"),
#     'rnet': os.path.join(WEIGHT_DIR, "Quant_Rnetv2_weight.pth"),
#     'onet': os.path.join(WEIGHT_DIR, "Quant_Onetv2_weight.pth")
# }

# # ---- Device ----
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # ---- Load models ----
# pnet = QuantPNet().to(device)
# rnet = QuantRNet().to(device)
# onet = QuantONet().to(device)
# for name, m in [('pnet', pnet), ('rnet', rnet), ('onet', onet)]:
#     wp = weight_paths[name]
#     if not os.path.exists(wp):
#         raise FileNotFoundError(f"Weight file not found: {wp}")
#     m.load_state_dict(torch.load(wp, map_location=device))
#     m.eval()

# # ---- Load image ----
# image_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/img/2.jpg"
# image = cv2.imread(image_path)
# if image is None:
#     raise FileNotFoundError(f"Could not load image: {image_path}")
# height, width, _ = image.shape
# print(f"Input image: {image.shape}")

# # ---- MTCNN params ----
# min_face_size   = 20.0
# thresholds      = [0.6, 0.6, 0.6]   # P, R, O thresholds
# nms_t           = [0.7, 0.8, 0.9]   # P, R, O NMS overlaps
# min_detect_size = 12
# scale_factor    = 0.707

# # ---- Helpers ----
# def preprocess(img):
#     # BGR→RGB, normalize to [-1,1], CHW, add batch
#     img = img[:, :, ::-1].astype(np.float32)
#     img = img.transpose(2,0,1)[None,...]
#     return (img - 127.5) * 0.0078125

# def to_numpy(qtensor):
#     return qtensor.value.detach().cpu().numpy() if hasattr(qtensor, "value") else qtensor.detach().cpu().numpy()

# def nms(boxes, thresh, mode="union"):
#     if boxes.shape[0] == 0:
#         return []
#     x1,y1,x2,y2,s = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
#     area = (x2-x1+1)*(y2-y1+1)
#     idxs = np.argsort(s)
#     pick = []
#     while idxs.size > 0:
#         i = idxs[-1]
#         pick.append(i)
#         xx1 = np.maximum(x1[i], x1[idxs[:-1]])
#         yy1 = np.maximum(y1[i], y1[idxs[:-1]])
#         xx2 = np.minimum(x2[i], x2[idxs[:-1]])
#         yy2 = np.minimum(y2[i], y2[idxs[:-1]])
#         w = np.maximum(0, xx2-xx1+1)
#         h = np.maximum(0, yy2-yy1+1)
#         inter = w*h
#         if mode == "min":
#             o = inter / np.minimum(area[i], area[idxs[:-1]])
#         else:
#             o = inter / (area[i] + area[idxs[:-1]] - inter)
#         idxs = np.delete(idxs,
#                          np.concatenate(([len(idxs)-1],
#                                          np.where(o > thresh)[0])))
#     return pick

# def calibrate_box(bboxes, offsets):
#     x1,y1,x2,y2 = [bboxes[:,i] for i in range(4)]
#     w = (x2-x1+1)[:,None]; h = (y2-y1+1)[:,None]
#     trans = np.hstack([w,h,w,h]) * offsets
#     bboxes[:,:4] += trans
#     return bboxes

# def to_square(bboxes):
#     sq = bboxes.copy()
#     x1,y1,x2,y2 = [sq[:,i] for i in range(4)]
#     h = y2-y1+1; w = x2-x1+1
#     m = np.maximum(h,w)
#     sq[:,0] = x1 + w*0.5 - m*0.5
#     sq[:,1] = y1 + h*0.5 - m*0.5
#     sq[:,2] = sq[:,0] + m - 1
#     sq[:,3] = sq[:,1] + m - 1
#     sq[:,4] = bboxes[:,4]
#     return sq

# def correct_bboxes(bboxes, W, H):
#     x1,y1,x2,y2 = [bboxes[:,i] for i in range(4)]
#     w = x2-x1+1; h = y2-y1+1
#     dx, dy = np.zeros_like(x1), np.zeros_like(y1)
#     edx, edy = w-1, h-1
#     ex, ey = x2.copy(), y2.copy()
#     # right overflow
#     idx = np.where(ex > W-1)[0]
#     edx[idx] = w[idx] + W - 2 - ex[idx]
#     ex[idx] = W - 1
#     # bottom overflow
#     idx = np.where(ey > H-1)[0]
#     edy[idx] = h[idx] + H - 2 - ey[idx]
#     ey[idx] = H - 1
#     # left overflow
#     idx = np.where(x1 < 0)[0]
#     dx[idx] = -x1[idx]; x1[idx] = 0
#     # top overflow
#     idx = np.where(y1 < 0)[0]
#     dy[idx] = -y1[idx]; y1[idx] = 0
#     return [dy, edy, dx, edx, y1, ey, x1, ex, w, h]

# # ==== Stage 1: P-Net ====
# scales = []
# m = min_detect_size / min_face_size
# min_len = min(height, width) * m
# while min_len > min_detect_size:
#     scales.append(m)
#     m *= scale_factor
#     min_len *= scale_factor

# all_boxes = []
# with torch.no_grad():
#     for scale in scales:
#         sw, sh = math.ceil(width*scale), math.ceil(height*scale)
#         im_s = cv2.resize(image, (sw, sh))
#         inp = torch.FloatTensor(preprocess(im_s)).to(device)
#         qt_bbox, qt_conf = pnet(inp)
#         probs   = to_numpy(qt_conf)[0,1,:,:]
#         offsets = to_numpy(qt_bbox)[0]
#         ys, xs = np.where(probs > thresholds[0])
#         if ys.size>0:
#             tx1, ty1, tx2, ty2 = [offsets[i, ys, xs] for i in range(4)]
#             sc = probs[ys, xs]
#             boxes = np.vstack([
#                 np.round((2*xs + 1) / scale),
#                 np.round((2*ys + 1) / scale),
#                 np.round((2*xs + 1 + 12) / scale),
#                 np.round((2*ys + 1 + 12) / scale),
#                 sc,
#                 tx1, ty1, tx2, ty2
#             ]).T
#             keep = nms(boxes[:,:5], nms_t[0])
#             all_boxes.append(boxes[keep])

# if not all_boxes:
#     print("P-Net found no faces"); exit()
# boxes = np.vstack(all_boxes)
# keep = nms(boxes[:,:5], nms_t[0]); boxes = boxes[keep]
# boxes = calibrate_box(boxes, boxes[:,5:])
# boxes = to_square(boxes)
# boxes[:,:4] = np.round(boxes[:,:4])
# print(f"P-Net → {boxes.shape[0]} boxes")

# p_img = image.copy()
# for bb in boxes:
#     x1,y1,x2,y2 = map(int, bb[:4])
#     cv2.rectangle(p_img, (x1,y1), (x2,y2), (0,0,255), 1)

# # ==== Stage 2: R-Net ====
# if boxes.shape[0] > 0:
#     size, N = 24, boxes.shape[0]
#     dy, edy, dx, edx, y1, ey, x1, ex, w_arr, h_arr = correct_bboxes(boxes, width, height)
#     # cast to int
#     dy, edy, dx, edx = [arr.astype(int) for arr in (dy, edy, dx, edx)]
#     y1, ey, x1, ex   = [arr.astype(int) for arr in (y1, ey, x1, ex)]
#     w_arr, h_arr     = w_arr.astype(int), h_arr.astype(int)

#     crops = np.zeros((N,3,size,size), dtype=np.float32)
#     for i in range(N):
#         if h_arr[i]<=0 or w_arr[i]<=0:
#             continue
#         patch = np.zeros((h_arr[i], w_arr[i], 3), dtype=np.uint8)
#         patch = image[y1[i]:ey[i]+1, x1[i]:ex[i]+1]
#         patch = cv2.resize(patch, (size, size))
#         crops[i] = preprocess(patch)[0]

#     inp2 = torch.FloatTensor(crops).to(device)
#     with torch.no_grad():
#         qt_bbox2, qt_conf2 = rnet(inp2)
#     probs2   = to_numpy(qt_conf2)[:,1]
#     offsets2 = to_numpy(qt_bbox2)          # shape (N,4)

#     idx = np.where(probs2 > thresholds[1])[0]
#     boxes    = boxes[idx]
#     boxes[:,4] = probs2[idx]
#     offsets2 = offsets2[idx]

#     keep = nms(boxes, nms_t[1])
#     boxes    = calibrate_box(boxes[keep], offsets2[keep])
#     offsets2 = offsets2[keep]
#     boxes    = to_square(boxes)
#     boxes[:,:4] = np.round(boxes[:,:4])
#     print(f"R-Net → {boxes.shape[0]} boxes")
# else:
#     print("R-Net skipped")

# r_img = image.copy()
# for bb in boxes:
#     x1,y1,x2,y2 = map(int, bb[:4])
#     cv2.rectangle(r_img, (x1,y1), (x2,y2), (0,255,0), 1)

# # ==== Stage 3: O-Net ====
# if boxes.shape[0] > 0:
#     size, N = 48, boxes.shape[0]

#     # rois are already square from R-Net
#     rois = boxes.copy()
#     rois[:, [0,2]] = np.clip(rois[:, [0,2]], 0, width-1)
#     rois[:, [1,3]] = np.clip(rois[:, [1,3]], 0, height-1)

#     crops = np.zeros((N,3,size,size), dtype=np.float32)
#     for i, bb in enumerate(rois):
#         x1, y1, x2, y2 = map(int, bb[:4])
#         if x2<=x1 or y2<=y1: continue
#         patch = cv2.resize(image[y1:y2+1, x1:x2+1], (size, size))
#         crops[i] = preprocess(patch)[0]

#     inp3 = torch.FloatTensor(crops).to(device)
#     with torch.no_grad():
#         qt_bbox3, qt_conf3, qt_lm3 = onet(inp3)

#     probs3   = to_numpy(qt_conf3)[:,1]
#     offsets3 = to_numpy(qt_bbox3)

#     idx = np.where(probs3 > thresholds[2])[0]
#     if idx.size == 0:
#         print("O-Net → 0 boxes")
#         o_img = image.copy()
#     else:
#         sel_rois    = rois[idx]
#         sel_offsets = offsets3[idx]

#         final_boxes = calibrate_box(sel_rois.copy(), sel_offsets)
#         final_boxes = to_square(final_boxes)
#         keep3 = nms(final_boxes, nms_t[2], mode='union')
#         final_boxes = final_boxes[keep3]
#         print(f"O-Net → {final_boxes.shape[0]} boxes")

#         o_img = image.copy()
#         for bb in final_boxes:
#             x1,y1,x2,y2 = map(int, bb[:4])
#             cv2.rectangle(o_img, (x1,y1), (x2,y2), (255,0,0), 2)
# else:
#     print("O-Net skipped")
#     o_img = image.copy()

# # ==== Show results ====
# plt.figure(figsize=(15,5))
# for img, title in [(p_img,"P-Net"), (r_img,"R-Net"), (o_img,"O-Net")]:
#     plt.subplot(1,3,["P-Net","R-Net","O-Net"].index(title)+1)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis('off')
# plt.tight_layout()
# plt.show()



import os
import math
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from Quantization_MTCNNv2 import QuantPNet, QuantRNet, QuantONet

# Windows OpenMP workaround
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Paths ----
WEIGHT_DIR = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights"
weight_paths = {
    'pnet': os.path.join(WEIGHT_DIR, "Quan_Pnetv2_weight.pth"),
    'rnet': os.path.join(WEIGHT_DIR, "Quant_Rnetv2_weight.pth"),
    'onet': os.path.join(WEIGHT_DIR, "Quant_Onetv2_weight.pth")
}

# ---- Device ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- Load models ----
pnet = QuantPNet().to(device)
rnet = QuantRNet().to(device)
onet = QuantONet().to(device)
for name, m in [('pnet', pnet), ('rnet', rnet), ('onet', onet)]:
    fp = weight_paths[name]
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{name} weights not found: {fp}")
    m.load_state_dict(torch.load(fp, map_location=device))
    m.eval()

# ---- Load image ----
img_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/img/2.jpg"
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(img_path)
H, W, _ = image.shape
print(f"Input image: {image.shape}")

# ---- Params ----
min_face_size   = 20.0
thresholds      = [0.6, 0.5, 0.1]   # P-Net, R-Net, O-Net thresholds
nms_t           = [0.7, 0.7, 0.9]   # NMS overlaps
min_det_size    = 12
scale_factor    = 0.707

# ---- Helpers ----
def preprocess(img):
    x = img[:, :, ::-1].astype(np.float32)   # BGR→RGB
    x = x.transpose(2,0,1)[None,...]         # CHW + batch
    return (x - 127.5)*0.0078125

def to_numpy(qt):
    return qt.value.detach().cpu().numpy() if hasattr(qt, "value") else qt.detach().cpu().numpy()

def nms(boxes, th, mode="union"):
    if boxes.shape[0]==0: return []
    x1,y1,x2,y2,s = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],boxes[:,4]
    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort(s)
    pick=[]
    while idxs.size>0:
        i = idxs[-1]; pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])
        w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
        inter = w*h
        if mode=="min":
            o = inter/np.minimum(area[i], area[idxs[:-1]])
        else:
            o = inter/(area[i]+area[idxs[:-1]]-inter)
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(o>th)[0])))
    return pick

def calibrate_box(bboxes, offs):
    x1,y1,x2,y2 = [bboxes[:,i] for i in range(4)]
    w = (x2-x1+1)[:,None]; h = (y2-y1+1)[:,None]
    trans = np.hstack([w,h,w,h]) * offs
    bboxes[:,:4] += trans
    return bboxes

def to_square(bb):
    sq = bb.copy()
    x1,y1,x2,y2 = [sq[:,i] for i in range(4)]
    h = y2-y1+1; w = x2-x1+1
    m = np.maximum(h,w)
    sq[:,0] = x1 + w*0.5 - m*0.5
    sq[:,1] = y1 + h*0.5 - m*0.5
    sq[:,2] = sq[:,0] + m - 1
    sq[:,3] = sq[:,1] + m - 1
    sq[:,4] = bb[:,4]
    return sq

def correct_bboxes(bboxes, W, H):
    x1,y1,x2,y2 = [bboxes[:,i] for i in range(4)]
    w = x2-x1+1; h = y2-y1+1
    dx, dy = np.zeros_like(x1), np.zeros_like(y1)
    edx, edy = w-1, h-1
    ex, ey = x2.copy(), y2.copy()
    idx = np.where(ex>W-1)[0]; edx[idx]=w[idx]+W-2-ex[idx]; ex[idx]=W-1
    idx = np.where(ey>H-1)[0]; edy[idx]=h[idx]+H-2-ey[idx]; ey[idx]=H-1
    idx = np.where(x1<0)[0]; dx[idx]=-x1[idx]; x1[idx]=0
    idx = np.where(y1<0)[0]; dy[idx]=-y1[idx]; y1[idx]=0
    return [dy, edy, dx, edx, y1, ey, x1, ex, w, h]

# ==== Stage 1: P-Net ====
scales=[]; m = min_det_size/min_face_size; min_len=min(H,W)*m
while min_len>min_det_size:
    scales.append(m); m*=scale_factor; min_len*=scale_factor

all_boxes=[]
with torch.no_grad():
    for sc in scales:
        sw,sh = math.ceil(W*sc), math.ceil(H*sc)
        sm = cv2.resize(image,(sw,sh))
        inp = torch.FloatTensor(preprocess(sm)).to(device)
        qt_b, qt_c = pnet(inp)
        probs = to_numpy(qt_c)[0,1,:,:]
        offs  = to_numpy(qt_b)[0]
        ys,xs = np.where(probs>thresholds[0])
        if ys.size>0:
            txy = [offs[i,ys,xs] for i in range(4)]
            scs = probs[ys,xs]
            bbs = np.vstack([
                np.round((2*xs+1)/sc),
                np.round((2*ys+1)/sc),
                np.round((2*xs+1+12)/sc),
                np.round((2*ys+1+12)/sc),
                scs,
                txy[0],txy[1],txy[2],txy[3]
            ]).T
            keep = nms(bbs[:,:5], nms_t[0])
            all_boxes.append(bbs[keep])
if not all_boxes:
    print("P-Net found nothing"); exit()
boxes = np.vstack(all_boxes)
keep = nms(boxes[:,:5], nms_t[0]); boxes=boxes[keep]
boxes = calibrate_box(boxes, boxes[:,5:])
boxes = to_square(boxes)
boxes[:,:4] = np.round(boxes[:,:4])
print(f"P-Net → {boxes.shape[0]}")

p_img=image.copy()
for bb in boxes:
    x1,y1,x2,y2=map(int,bb[:4])
    cv2.rectangle(p_img,(x1,y1),(x2,y2),(0,0,255),1)

# ==== Stage 2: R-Net ====
if boxes.shape[0]>0:
    size,N=24,boxes.shape[0]
    dy,edy,dx,edx,y1,ey,x1,ex,w_arr,h_arr = correct_bboxes(boxes,W,H)
    dy,edy,dx,edx=[a.astype(int) for a in (dy,edy,dx,edx)]
    y1,ey,x1,ex=[a.astype(int) for a in (y1,ey,x1,ex)]
    w_arr,h_arr = w_arr.astype(int),h_arr.astype(int)

    crops=np.zeros((N,3,size,size),np.float32)
    for i in range(N):
        if h_arr[i]<=0 or w_arr[i]<=0: continue
        patch=image[y1[i]:ey[i]+1,x1[i]:ex[i]+1]
        patch=cv2.resize(patch,(size,size))
        crops[i]=preprocess(patch)[0]

    inp2=torch.FloatTensor(crops).to(device)
    with torch.no_grad():
        qt_b2, qt_c2 = rnet(inp2)
    p2 = to_numpy(qt_c2)[:,1]
    o2 = to_numpy(qt_b2)
    idx = np.where(p2>thresholds[1])[0]
    boxes    = boxes[idx]
    boxes[:,4]=p2[idx]
    o2       = o2[idx]
    keep     = nms(boxes, nms_t[1])
    boxes    = calibrate_box(boxes[keep], o2[keep])
    o2       = o2[keep]
    boxes    = to_square(boxes)
    boxes[:,:4]=np.round(boxes[:,:4])
    print(f"R-Net → {boxes.shape[0]}")
else:
    print("R-Net skipped")

r_img=image.copy()
for bb in boxes:
    x1,y1,x2,y2=map(int,bb[:4])
    cv2.rectangle(r_img,(x1,y1),(x2,y2),(0,255,0),1)

# ==== Stage 3: O-Net ====
if boxes.shape[0]>0:
    size,N=48,boxes.shape[0]
    rois=boxes.copy()
    rois[:,[0,2]]=np.clip(rois[:,[0,2]],0,W-1)
    rois[:,[1,3]]=np.clip(rois[:,[1,3]],0,H-1)

    crops=np.zeros((N,3,size,size),np.float32)
    for i,bb in enumerate(rois):
        x1,y1,x2,y2=map(int,bb[:4])
        if x2<=x1 or y2<=y1: continue
        patch=cv2.resize(image[y1:y2+1,x1:x2+1],(size,size))
        crops[i]=preprocess(patch)[0]

    inp3=torch.FloatTensor(crops).to(device)
    with torch.no_grad():
        qt_lm3, qt_b3, qt_c3 = onet(inp3)

    p3 = to_numpy(qt_c3)[:,1]
    o3 = to_numpy(qt_b3)
    print("ONet confidences:", np.round(p3,3), "max:", p3.max())

    idx = np.where(p3>thresholds[2])[0]
    if idx.size==0:
        print("O-Net → 0")
        o_img=image.copy()
    else:
        sel_rois    = rois[idx]
        sel_offsets = o3[idx]
        fb = calibrate_box(sel_rois.copy(), sel_offsets)
        fb = to_square(fb)
        keep3 = nms(fb, nms_t[2], mode="union")
        fb    = fb[keep3]
        print(f"O-Net → {fb.shape[0]}")

        o_img=image.copy()
        for bb in fb:
            x1,y1,x2,y2=map(int,bb[:4])
            cv2.rectangle(o_img,(x1,y1),(x2,y2),(255,0,0),2)
else:
    print("O-Net skipped")
    o_img=image.copy()

# ==== Display ====
plt.figure(figsize=(15,5))
for img,title in [(p_img,"P-Net"),(r_img,"R-Net"),(o_img,"O-Net")]:
    plt.subplot(1,3,["P-Net","R-Net","O-Net"].index(title)+1)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(title); plt.axis("off")
plt.tight_layout()
plt.show()


import os
import time
import copy
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from Quantization_MTCNNv2 import QuantPNet  

# Dataset
class ListDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as f:
            self.img_files = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        parts = self.img_files[idx].split()
        img_path = parts[0]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) * 0.0078125   # normalize
        img = img.transpose(2, 0, 1)      # HWC → CHW
        input_tensor = torch.from_numpy(img)

        label = int(parts[1])
        bbox = np.zeros(4, dtype=np.float32)
        if label != 0 and len(parts) == 6:
            bbox = np.array([float(x) for x in parts[2:]], dtype=np.float32)
        elif label != 0:
            print(f"Warning: invalid annotation at {idx}: {parts}")

        return {
            'input_img': input_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'bbox_target': torch.from_numpy(bbox)
        }

# Weight init (chỉ ảnh hưởng các layer PyTorch gốc, Quant layers sẽ sử dụng STE cho weight)
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)

def main():
    # Paths
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_val.txt"
    batch_size = 32
    num_epochs = 15

    # Datasets & Dataloaders
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: annotation file not found.")
        return

    dataloaders = {
        phase: DataLoader(
            ListDataset(path),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        for phase, path in [('train', train_path), ('val', val_path)]
    }
    dataset_sizes = {ph: len(dataloaders[ph].dataset) for ph in dataloaders}
    print(f"Dataset sizes: {dataset_sizes}")

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = QuantPNet().to(device)
    model.apply(weights_init)

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_cls    = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    since = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = running_cls = running_off = 0.0
            running_correct = running_gt = 0

            for batch in dataloaders[phase]:
                imgs   = batch['input_img'].to(device)
                gt_lbl = batch['label'].to(device)
                gt_off = batch['bbox_target'].to(device).float()

                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    pred_off, pred_lbl = model(imgs)
                    # pred_* là QuantTensor → lấy .value
                    pred_lbl = pred_lbl.value       # [B, 2, H, W]
                    pred_off = pred_off.value      # [B, 4, H, W]
                    # global pooling (mean)
                    pred_lbl = pred_lbl.mean(dim=[2,3])
                    pred_off = pred_off.mean(dim=[2,3])

                    # classification loss
                    mask_cls = (gt_lbl >= 0)
                    valid_lbl = gt_lbl[mask_cls]
                    valid_pred_lbl = pred_lbl[mask_cls]
                    cls_l = torch.tensor(0.0, device=device)
                    correct = 0
                    if valid_lbl.numel() > 0:
                        cls_l = loss_cls(valid_pred_lbl, valid_lbl)
                        preds = valid_pred_lbl.argmax(dim=1)
                        correct = (preds == valid_lbl).sum().item()

                    # offset loss
                    mask_off = (gt_lbl != 0)
                    valid_off = gt_off[mask_off]
                    valid_pred_off = pred_off[mask_off]
                    off_l = torch.tensor(0.0, device=device)
                    if valid_off.numel() > 0:
                        off_l = loss_offset(valid_pred_off, valid_off)

                    # combined
                    loss = 0.02*cls_l + 0.6*off_l if (cls_l>0 or off_l>0) else cls_l

                    if is_train:
                        loss.backward()
                        optimizer.step()

                # stats
                bsz = imgs.size(0)
                running_loss += loss.item() * bsz
                running_cls  += cls_l.item() * bsz
                running_off  += off_l.item() * bsz
                running_correct += correct
                running_gt += valid_lbl.numel()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_correct / (running_gt+1e-8)
            print(f"{phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase=='val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val loss: {best_loss:.4f}")

    # Save best weights
    model.load_state_dict(best_wts)
    save_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Quan_Pnetv2_weight.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    main()

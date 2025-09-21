import sys
sys.path.append('..')
import os
import time
import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Quantization_MTCNNv2 import QuantRNet  # model mới không cần is_train


# Dataset class
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
        img = (img - 127.5) * 0.0078125
        img = img.transpose(2, 0, 1)
        input_img = torch.from_numpy(img)

        label = int(parts[1])
        bbox_target = np.zeros(4, dtype=np.float32)
        landmark = np.zeros(10, dtype=np.float32)

        # annotation có 6 phần tử → chỉ bbox
        if len(parts) == 6:
            bbox_target = np.array(parts[2:6], dtype=np.float32)
        # annotation có 14 phần tử → bbox + 5 điểm landmark
        elif len(parts) == 14:
            bbox_target = np.array(parts[2:6], dtype=np.float32)
            landmark    = np.array(parts[6:], dtype=np.float32)

        return {
            'input_img':    input_img,
            'label':        torch.tensor(label, dtype=torch.long),
            'bbox_target':  torch.from_numpy(bbox_target),
            'landmark':     torch.from_numpy(landmark)
        }


# Weight init (áp dụng cho các layer PyTorch gốc)
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)


def train_model(model, dataloaders, dataset_sizes, device, num_epochs=20):
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc  = 0.0

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_cls    = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_correct = 0
            running_gt = 0

            for batch in dataloaders[phase]:
                imgs   = batch['input_img'].to(device)
                gt_lbl = batch['label'].to(device)
                gt_off = batch['bbox_target'].float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    pred_off_q, pred_lbl_q = model(imgs)
                    # Lấy ra Tensor từ QTensor
                    pred_off = pred_off_q.value  # [B,4]
                    pred_lbl = pred_lbl_q.value  # [B,2]

                    # Classification
                    mask_cls = (gt_lbl >= 0)
                    valid_gt_lbl = gt_lbl[mask_cls]
                    valid_pred_lbl = pred_lbl[mask_cls]
                    cls_l = torch.tensor(0.0, device=device)
                    correct = 0
                    if valid_gt_lbl.numel() > 0:
                        cls_l = loss_cls(valid_pred_lbl, valid_gt_lbl)
                        preds = valid_pred_lbl.argmax(dim=1)
                        correct = (preds == valid_gt_lbl).sum().item()

                    # Offset regression (chỉ positive và part, label ≠ 0)
                    mask_off = (gt_lbl != 0)
                    valid_gt_off   = gt_off[mask_off]
                    valid_pred_off = pred_off[mask_off]
                    off_l = torch.tensor(0.0, device=device)
                    if valid_gt_off.numel() > 0:
                        off_l = loss_offset(valid_pred_off, valid_gt_off)

                    # Tổng loss
                    loss = 0.02 * cls_l + 0.6 * off_l if (cls_l>0 or off_l>0) else cls_l

                    if is_train:
                        loss.backward()
                        optimizer.step()

                bsz = imgs.size(0)
                running_loss += loss.item() * bsz
                running_correct += correct
                running_gt += valid_gt_lbl.numel()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_correct / (running_gt + 1e-8)
            print(f'{phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # checkpoint
            if phase=='val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc  = epoch_acc
                best_wts  = copy.deepcopy(model.state_dict())

    total_time = time.time() - since
    print(f'\nTraining complete in {total_time//60:.0f}m {total_time%60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}, Best val acc: {best_acc:.4f}')

    model.load_state_dict(best_wts)
    return model


def main():
    # Paths
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_24_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_24_val.txt"
    batch_size= 32

    # DataLoaders
    dataloaders = {
        'train': DataLoader(ListDataset(train_path), batch_size=batch_size,
                            shuffle=True,  num_workers=4, pin_memory=True),
        'val':   DataLoader(ListDataset(val_path),   batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    }
    dataset_sizes = {
        ph: len(dataloaders[ph].dataset)
        for ph in ['train','val']
    }
    print(f"Dataset sizes: {dataset_sizes}")

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = QuantRNet().to(device)
    model.apply(weights_init)

    # Train
    model = train_model(model, dataloaders, dataset_sizes, device, num_epochs=20)

    # Save
    save_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Quant_Rnetv2_weight.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")


if __name__ == '__main__':
    main()

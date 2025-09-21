import sys, os, time, copy, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Quantization_MTCNNv2 import QuantONet

# -------- Dataset --------
class ListDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as f:
            self.lines = [l.strip() for l in f if l.strip()]
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        parts = self.lines[idx].split()
        img = cv2.imread(parts[0])
        if img is None:
            raise FileNotFoundError(f"{parts[0]} not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) * 0.0078125
        img = img.transpose(2,0,1)
        sample = {'input_img': torch.from_numpy(img)}
        sample['label'] = torch.tensor(int(parts[1]), dtype=torch.long)
        # bbox
        bbox = np.zeros(4, dtype=np.float32)
        if len(parts) >= 6:
            bbox = np.array(parts[2:6], dtype=np.float32)
        sample['bbox_target'] = torch.from_numpy(bbox)
        # landmark
        lm = np.zeros(10, dtype=np.float32)
        if len(parts) == 14:
            lm = np.array(parts[6:], dtype=np.float32)
        sample['landmark'] = torch.from_numpy(lm)
        return sample

# -------- weight init --------
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)

def main():
    # -------- hyperparams & data --------
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_48_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_48_val.txt"
    batch_size = 32
    num_epochs = 25
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloaders = {
        'train': DataLoader(ListDataset(train_path),
                             batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True),
        'val':   DataLoader(ListDataset(val_path),
                             batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    }
    dataset_sizes = {ph: len(dataloaders[ph].dataset) for ph in dataloaders}

    # -------- model, loss, optimizer --------
    model = QuantONet().to(device)
    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_cls      = nn.CrossEntropyLoss()
    loss_offset   = nn.MSELoss()
    loss_landmark = nn.MSELoss()

    # -------- training loop --------
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc  = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*30)
        for phase in ['train','val']:
            is_train = (phase=='train')
            model.train() if is_train else model.eval()

            running_loss = running_correct = running_gt = 0.0

            for batch in dataloaders[phase]:
                imgs   = batch['input_img'].to(device)
                gt_lbl = batch['label'].to(device)
                gt_off = batch['bbox_target'].float().to(device)
                gt_lm  = batch['landmark'].float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    qt_bbox, qt_conf, qt_landmark = model(imgs)
                    pred_off  = qt_bbox.value
                    pred_conf = qt_conf.value
                    pred_lm   = qt_landmark.value

                    # classification
                    mask_cls = (gt_lbl >= 0)
                    vgt_lbl = gt_lbl[mask_cls]
                    vp_conf = pred_conf[mask_cls]
                    cls_l = torch.tensor(0.0, device=device)
                    correct = 0
                    if vgt_lbl.numel()>0:
                        cls_l = loss_cls(vp_conf, vgt_lbl)
                        preds = vp_conf.argmax(dim=1)
                        correct = (preds==vgt_lbl).sum().item()

                    # offset
                    mask_off = (gt_lbl != 0)
                    voff = gt_off[mask_off]; poff = pred_off[mask_off]
                    off_l = torch.tensor(0.0, device=device)
                    if voff.numel()>0:
                        off_l = loss_offset(poff, voff)

                    # landmark
                    mask_lm = (gt_lm.abs().sum(dim=1)>0)
                    vlm = gt_lm[mask_lm]; plm = pred_lm[mask_lm]
                    lm_l = torch.tensor(0.0, device=device)
                    if vlm.numel()>0:
                        lm_l = loss_landmark(plm, vlm)

                    loss = 0.02*cls_l + 0.6*off_l + 3.0*lm_l
                    if is_train:
                        loss.backward()
                        optimizer.step()

                bsz = imgs.size(0)
                running_loss    += loss.item()*bsz
                running_correct += correct
                running_gt      += vgt_lbl.numel()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_correct / (running_gt+1e-8)
            print(f"{phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase=='val' and epoch_loss<best_loss:
                best_loss = epoch_loss
                best_acc  = epoch_acc
                best_wts  = copy.deepcopy(model.state_dict())

    elapsed = time.time()-since
    print(f"\nTraining complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Best val Loss: {best_loss:.4f}, Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_wts)
    save_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Quant_Onetv2_weight.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

if __name__ == '__main__':
    # cần cho Windows multiprocessing spawn
    from multiprocessing import freeze_support
    freeze_support()
    main()

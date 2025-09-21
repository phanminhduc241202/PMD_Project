import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import time
import copy
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from MTCNN_nets import PNet

# Dataset loader
class ListDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = [line.strip() for line in file if line.strip()]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        annotation = self.img_files[index].split()
        img_path = annotation[0]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found")
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)
        
        label = int(annotation[1])
        bbox_target = np.zeros(4, dtype=np.float32)
        if label != 0 and len(annotation) == 6:
            bbox_target = np.array([float(x) for x in annotation[2:]], dtype=np.float32)
        
        return {
            'input_img': input_img,
            'label': label,
            'bbox_target': bbox_target
        }

# Weight init
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

def main():
    # Path
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_val.txt"
    batch_size = 32
    num_epochs = 20

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Annotation file not found!")
        return

    dataloaders = {
        'train': DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    }
    dataset_sizes = {'train': len(ListDataset(train_path)), 'val': len(ListDataset(val_path))}
    print(f"Dataset sizes: train={dataset_sizes['train']}, val={dataset_sizes['val']}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PNet(is_train=True).to(device)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters())
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()

    # Logs
    train_loss_log, val_loss_log = [], []
    train_cls_loss_log, val_cls_loss_log = [], []
    train_offset_loss_log, val_offset_loss_log = [], []
    train_acc_log, val_acc_log = [], []

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs-1}')
        print('-' * 20)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = running_loss_cls = running_loss_offset = running_correct = running_gt = 0.0

            for sample_batched in dataloaders[phase]:
                inputs = sample_batched['input_img'].to(device)
                gt_label = sample_batched['label'].to(device)
                gt_offset = sample_batched['bbox_target'].to(device).float()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred_offset, pred_label = model(inputs)
                    pred_label = pred_label.mean(dim=[2, 3])
                    pred_offset = pred_offset.mean(dim=[2, 3])

                    mask_cls = gt_label >= 0
                    valid_gt_label = gt_label[mask_cls]
                    valid_pred_label = pred_label[mask_cls]

                    cls_loss = torch.tensor(0.0, device=device)
                    eval_correct = 0
                    num_gt = len(valid_gt_label)
                    if num_gt > 0:
                        cls_loss = loss_cls(valid_pred_label, valid_gt_label)
                        pred = torch.max(valid_pred_label, 1)[1]
                        eval_correct = (pred == valid_gt_label).sum().item()

                    mask_offset = gt_label != 0
                    valid_gt_offset = gt_offset[mask_offset]
                    valid_pred_offset = pred_offset[mask_offset]

                    offset_loss = torch.tensor(0.0, device=device)
                    if len(valid_gt_offset) > 0:
                        offset_loss = loss_offset(valid_pred_offset, valid_gt_offset)

                    loss = 0.02 * cls_loss + 0.6 * offset_loss if (cls_loss > 0 or offset_loss > 0) else cls_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_loss_cls += cls_loss.item() * inputs.size(0) if cls_loss > 0 else 0
                running_loss_offset += offset_loss.item() * inputs.size(0) if offset_loss > 0 else 0
                running_correct += eval_correct
                running_gt += num_gt

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_cls_loss = running_loss_cls / dataset_sizes[phase]
            epoch_offset_loss = running_loss_offset / dataset_sizes[phase]
            epoch_acc = running_correct / (running_gt + 1e-16)

            print(f'{phase} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
                  f'Cls Loss: {epoch_cls_loss:.4f}, Offset Loss: {epoch_offset_loss:.4f}')

            if phase == 'train':
                train_loss_log.append(epoch_loss)
                train_cls_loss_log.append(epoch_cls_loss)
                train_offset_loss_log.append(epoch_offset_loss)
                train_acc_log.append(epoch_acc)
            else:
                val_loss_log.append(epoch_loss)
                val_cls_loss_log.append(epoch_cls_loss)
                val_offset_loss_log.append(epoch_offset_loss)
                val_acc_log.append(epoch_acc)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

    # Save best model
    model.load_state_dict(best_model_wts)
    save_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Pnet_weight_v2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Model weights saved to {save_path}")

    # Plot
    epochs = list(range(num_epochs))
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_log, label='Train Loss')
    plt.plot(epochs, val_loss_log, label='Val Loss')
    plt.title('Total Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_cls_loss_log, label='Train Cls Loss')
    plt.plot(epochs, val_cls_loss_log, label='Val Cls Loss')
    plt.title('Classification Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_offset_loss_log, label='Train Offset Loss')
    plt.plot(epochs, val_offset_loss_log, label='Val Offset Loss')
    plt.title('Offset Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_acc_log, label='Train Accuracy')
    plt.plot(epochs, val_acc_log, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/plots/pnet_training_curves.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    main()

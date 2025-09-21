import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import time
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict
from MTCNN_nets import RNet

# Dataset class
class ListDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        annotation = self.img_files[index].strip().split()
        img_path = annotation[0]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)

        label = int(annotation[1])
        bbox_target = np.zeros(4, dtype=np.float32)
        if len(annotation[2:]) >= 4:
            bbox_target = np.array(annotation[2:6], dtype=np.float32)

        return {
            'input_img': input_img,
            'label': label,
            'bbox_target': bbox_target
        }

# # Flatten
# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# # RNet architecture
# class RNet(nn.Module):
#     def __init__(self, is_train=False):
#         super(RNet, self).__init__()
#         self.is_train = is_train

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 28, 3, 1), nn.PReLU(28),
#             nn.MaxPool2d(3, 2, ceil_mode=True),
#             nn.Conv2d(28, 48, 3, 1), nn.PReLU(48),
#             nn.MaxPool2d(3, 2, ceil_mode=True),
#             nn.Conv2d(48, 64, 2, 1), nn.PReLU(64),
#             Flatten(),
#             nn.Linear(576, 128), nn.PReLU(128)
#         )

#         self.classifier = nn.Linear(128, 2)
#         self.bbox_reg = nn.Linear(128, 4)

#     def forward(self, x):
#         x = self.features(x)
#         cls = self.classifier(x)
#         bbox = self.bbox_reg(x)
#         if not self.is_train:
#             cls = nn.functional.softmax(cls, dim=1)
#         return bbox, cls

# Weight init
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)

# Main training function
def main():
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_24_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_24_val.txt"
    weight_save_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Rnet_weight_v2.pth"
    plot_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/plots/rnet_training_curves.png"
    num_epochs = 20
    batch_size = 32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = {
        'train': DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type == 'cuda')),
        'val': DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == 'cuda'))
    }
    dataset_sizes = {k: len(v.dataset) for k, v in dataloaders.items()}
    print(f"Dataset sizes: train={dataset_sizes['train']}, val={dataset_sizes['val']}")

    model = RNet(is_train=True).to(device)
    model.apply(weights_init)

    optimizer = optim.Adam(model.parameters())
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()

    # Logs
    train_loss_log, val_loss_log = [], []
    train_cls_loss_log, val_cls_loss_log = [], []
    train_offset_loss_log, val_offset_loss_log = [], []
    train_acc_log, val_acc_log = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}\n" + "-"*20)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = running_cls = running_offset = running_correct = running_gt = 0.0

            for batch in dataloaders[phase]:
                inputs = batch['input_img'].to(device)
                labels = batch['label'].to(device)
                offsets = batch['bbox_target'].to(device).float()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred_offset, pred_label = model(inputs)

                    # Classification loss
                    mask_cls = labels >= 0
                    valid_labels = labels[mask_cls]
                    valid_preds = pred_label[mask_cls]
                    cls_loss = torch.tensor(0.0).to(device)
                    correct = 0
                    if valid_labels.numel() > 0:
                        cls_loss = loss_cls(valid_preds, valid_labels)
                        pred_classes = torch.argmax(valid_preds, dim=1)
                        correct = (pred_classes == valid_labels).sum().item()

                    # Offset loss
                    mask_offset = labels != 0
                    valid_offset = offsets[mask_offset]
                    pred_offset = pred_offset[mask_offset]
                    offset_loss = torch.tensor(0.0).to(device)
                    if valid_offset.numel() > 0:
                        offset_loss = loss_offset(pred_offset, valid_offset)

                    loss = 0.02 * cls_loss + 0.6 * offset_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_cls += cls_loss.item() * inputs.size(0)
                running_offset += offset_loss.item() * inputs.size(0)
                running_correct += correct
                running_gt += valid_labels.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_cls = running_cls / dataset_sizes[phase]
            epoch_offset = running_offset / dataset_sizes[phase]
            epoch_acc = running_correct / (running_gt + 1e-16)

            print(f"{phase.upper()} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
                  f"Cls Loss: {epoch_cls:.4f}, Offset Loss: {epoch_offset:.4f}")

            if phase == 'train':
                train_loss_log.append(epoch_loss)
                train_cls_loss_log.append(epoch_cls)
                train_offset_loss_log.append(epoch_offset)
                train_acc_log.append(epoch_acc)
            else:
                val_loss_log.append(epoch_loss)
                val_cls_loss_log.append(epoch_cls)
                val_offset_loss_log.append(epoch_offset)
                val_acc_log.append(epoch_acc)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

    # Save best weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), weight_save_path)
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Model weights saved to {weight_save_path}")

    # Plot results
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
    plt.savefig(plot_path)
    plt.show()
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    main()

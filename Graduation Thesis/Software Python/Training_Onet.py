import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2
from MTCNN_nets import ONet

# Dataset class
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        annotation = self.img_files[index % len(self.img_files)].strip().split(' ')
        img = cv2.imread(annotation[0])
        if img is None:
            raise ValueError(f"Failed to load image: {annotation[0]}")
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.asarray(img, 'float32').transpose((2, 0, 1))
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)

        label = int(annotation[1])
        bbox_target = np.zeros((4,), dtype=np.float32)
        landmark = np.zeros((10,), dtype=np.float32)

        if len(annotation[2:]) == 4:
            bbox_target = np.array(annotation[2:6], dtype=np.float32)
        elif len(annotation[2:]) == 14:
            bbox_target = np.array(annotation[2:6], dtype=np.float32)
            landmark = np.array(annotation[6:], dtype=np.float32)

        return {'input_img': input_img, 'label': label, 'bbox_target': bbox_target, 'landmark': landmark}

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# class ONet(nn.Module):
#     def __init__(self, is_train=False):
#         super(ONet, self).__init__()
#         self.is_train = is_train
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(3, 32, 3, 1)),
#             ('prelu1', nn.PReLU(32)),
#             ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
#             ('conv2', nn.Conv2d(32, 64, 3, 1)),
#             ('prelu2', nn.PReLU(64)),
#             ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
#             ('conv3', nn.Conv2d(64, 64, 3, 1)),
#             ('prelu3', nn.PReLU(64)),
#             ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
#             ('conv4', nn.Conv2d(64, 128, 2, 1)),
#             ('prelu4', nn.PReLU(128)),
#             ('flatten', Flatten()),
#             ('fc5', nn.Linear(1152, 256)),
#             ('drop5', nn.Dropout(0.25)),
#             ('prelu5', nn.PReLU(256)),
#         ]))
#         self.conv6_1 = nn.Linear(256, 2)
#         self.conv6_2 = nn.Linear(256, 4)
#         self.conv6_3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = self.features(x)
#         a = self.conv6_1(x)
#         b = self.conv6_2(x)
#         c = self.conv6_3(x)
#         if not self.is_train:
#             a = torch.softmax(a, dim=1)
#         return c, b, a

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

def tensor2list(log):
    return [v.item() if isinstance(v, torch.Tensor) else float(v) for v in log]

def main():
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_48_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_48_val.txt"
    weight_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Onet_weight_v2.pth"
    plot_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/plots/onet_training_curves.png"

    batch_size = 32
    num_epochs = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloaders = {
        'train': DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True),
        'val': DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True)
    }
    dataset_sizes = {'train': len(ListDataset(train_path)), 'val': len(ListDataset(val_path))}

    model = ONet(is_train=True).to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters())
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()
    loss_landmark = nn.MSELoss()

    # Logs
    train_loss_log, val_loss_log = [], []
    train_cls_loss_log, val_cls_loss_log = [], []
    train_offset_loss_log, val_offset_loss_log = [], []
    train_landmark_loss_log, val_landmark_loss_log = [], []
    train_acc_log, val_acc_log = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}\n{'-'*20}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = running_loss_cls = running_loss_offset = running_loss_landmark = running_correct = running_gt = 0.0

            for batch in dataloaders[phase]:
                input_images = batch['input_img'].to(device)
                gt_label = batch['label'].to(device)
                gt_offset = batch['bbox_target'].to(device)
                gt_landmark = batch['landmark'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    pred_lm, pred_offset, pred_label = model(input_images)
                    mask_cls = gt_label >= 0
                    valid_gt_label = gt_label[mask_cls]
                    valid_pred_label = pred_label[mask_cls]

                    mask_offset = gt_label != 0
                    valid_gt_offset = gt_offset[mask_offset]
                    valid_pred_offset = pred_offset[mask_offset]

                    mask_lm = gt_label == -2
                    valid_gt_lm = gt_landmark[mask_lm]
                    valid_pred_lm = pred_lm[mask_lm]

                    loss = torch.tensor(0.0, device=device)
                    cls_loss = offset_loss = landmark_loss = 0.0
                    eval_correct = 0
                    num_gt = len(valid_gt_label)

                    if num_gt > 0:
                        cls_loss = loss_cls(valid_pred_label, valid_gt_label)
                        loss += 0.02 * cls_loss
                        pred = torch.argmax(valid_pred_label, dim=1)
                        eval_correct = (pred == valid_gt_label).sum().item()

                    if len(valid_gt_offset) > 0:
                        offset_loss = loss_offset(valid_pred_offset, valid_gt_offset)
                        loss += 0.6 * offset_loss

                    if len(valid_gt_lm) > 0:
                        landmark_loss = loss_landmark(valid_pred_lm, valid_gt_lm)
                        loss += 3.0 * landmark_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                bs = input_images.size(0)
                running_loss += loss.item() * bs
                running_loss_cls += float(cls_loss) * bs if num_gt > 0 else 0
                running_loss_offset += float(offset_loss) * bs if len(valid_gt_offset) > 0 else 0
                running_loss_landmark += float(landmark_loss) * bs if len(valid_gt_lm) > 0 else 0
                running_correct += eval_correct
                running_gt += num_gt

            ep_loss = running_loss / dataset_sizes[phase]
            ep_cls_loss = running_loss_cls / dataset_sizes[phase]
            ep_offset_loss = running_loss_offset / dataset_sizes[phase]
            ep_landmark_loss = running_loss_landmark / dataset_sizes[phase]
            ep_acc = running_correct / (running_gt + 1e-4)

            print(f"{phase} Loss: {ep_loss:.4f}, Acc: {ep_acc:.4f}, Cls Loss: {ep_cls_loss:.4f}, "
                  f"Offset Loss: {ep_offset_loss:.4f}, Landmark Loss: {ep_landmark_loss:.4f}")

            # Logging
            if phase == 'train':
                train_loss_log.append(ep_loss)
                train_cls_loss_log.append(ep_cls_loss)
                train_offset_loss_log.append(ep_offset_loss)
                train_landmark_loss_log.append(ep_landmark_loss)
                train_acc_log.append(ep_acc)
            else:
                val_loss_log.append(ep_loss)
                val_cls_loss_log.append(ep_cls_loss)
                val_offset_loss_log.append(ep_offset_loss)
                val_landmark_loss_log.append(ep_landmark_loss)
                val_acc_log.append(ep_acc)
                if ep_loss < best_loss:
                    best_loss = ep_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), weight_path)
    print(f"Best val loss: {best_loss:.4f}")

    # Plotting
    epochs = list(range(num_epochs))
    plt.figure(figsize=(14, 12))
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_loss_log, label='Train Loss')
    plt.plot(epochs, val_loss_log, label='Val Loss')
    plt.title('Total Loss'); plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_cls_loss_log, label='Train Cls Loss')
    plt.plot(epochs, val_cls_loss_log, label='Val Cls Loss')
    plt.title('Classification Loss'); plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_offset_loss_log, label='Train Offset Loss')
    plt.plot(epochs, val_offset_loss_log, label='Val Offset Loss')
    plt.title('Offset Loss'); plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_landmark_loss_log, label='Train Landmark Loss')
    plt.plot(epochs, val_landmark_loss_log, label='Val Landmark Loss')
    plt.title('Landmark Loss'); plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(epochs, train_acc_log, label='Train Acc')
    plt.plot(epochs, val_acc_log, label='Val Acc')
    plt.title('Accuracy'); plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == '__main__':
    main()

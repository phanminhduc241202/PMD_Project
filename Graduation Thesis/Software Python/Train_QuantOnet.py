import sys
import os
sys.path.append('..')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import copy
import numpy as np
from collections import OrderedDict
import cv2
from Quantization_MTCNN import QuantONet
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  

# Dataset class for loading MTCNN data
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        annotation = self.img_files[index % len(self.img_files)].strip().split(' ')

        # Image loading and preprocessing
        img = cv2.imread(annotation[0])
        if img is None:
            raise ValueError(f"Failed to load image: {annotation[0]}")
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = (img - 127.5) * 0.0078125  # Normalize to [-1, 1]
        input_img = torch.FloatTensor(img)

        # Label and targets
        label = int(annotation[1])
        bbox_target = np.zeros((4,), dtype=np.float32)
        landmark = np.zeros((10,), dtype=np.float32)

        if len(annotation[2:]) == 4:
            bbox_target = np.array(annotation[2:6], dtype=np.float32)
        elif len(annotation[2:]) == 14:
            bbox_target = np.array(annotation[2:6], dtype=np.float32)
            landmark = np.array(annotation[6:], dtype=np.float32)

        sample = {'input_img': input_img, 'label': label, 'bbox_target': bbox_target, 'landmark': landmark}
        return sample

# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

# Training configuration
train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_48_train.txt"
val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_48_val.txt"
batch_size = 32
num_epochs = 25
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data loaders
try:
    dataloaders = {
        'train': DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True, num_workers=0)
    }
    dataset_sizes = {
        'train': len(ListDataset(train_path)),
        'val': len(ListDataset(val_path))
    }
except FileNotFoundError as e:
    print(f"Error: Annotation file not found: {e}")
    sys.exit(1)

# Initialize model
model = QuantONet(is_train=True).to(device)
model.apply(weights_init)

# Optimizer and loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_cls = nn.CrossEntropyLoss()
loss_offset = nn.MSELoss()
loss_landmark = nn.MSELoss()

# Training loop
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
best_accuracy = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs-1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_loss_cls = 0.0
        running_loss_offset = 0.0
        running_loss_landmark = 0.0
        running_correct = 0.0
        running_gt = 0.0

        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            input_images = sample_batched['input_img'].to(device)
            gt_label = sample_batched['label'].to(device)
            gt_offset = sample_batched['bbox_target'].type(torch.FloatTensor).to(device)
            landmark_offset = sample_batched['landmark'].type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                pred_landmark, pred_offsets, pred_label = model(input_images)
                # Trích xuất tensor từ IntQuantTensor
                pred_landmark = pred_landmark.value  # [batch_size, 10]
                pred_offsets = pred_offsets.value    # [batch_size, 4]
                pred_label = pred_label.value        # [batch_size, 2]

                # Classification loss (for label >= 0, i.e., positive and negative samples)
                mask_cls = torch.ge(gt_label, 0)
                valid_gt_label = gt_label[mask_cls]
                valid_pred_label = pred_label[mask_cls]

                # Bounding box loss (for label != 0, i.e., positive and part samples)
                unmask = torch.eq(gt_label, 0)
                mask_offset = torch.logical_not(unmask)
                valid_gt_offset = gt_offset[mask_offset]
                valid_pred_offset = pred_offsets[mask_offset]

                # Landmark loss (for samples with non-zero landmarks, typically label = 1)
                mask_lm = torch.any(landmark_offset != 0, dim=1)  # Samples with non-zero landmarks
                valid_landmark_offset = landmark_offset[mask_lm]
                valid_pred_landmark = pred_landmark[mask_lm]

                # Compute losses
                loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)
                offset_loss = torch.tensor(0.0, device=device)
                landmark_loss = torch.tensor(0.0, device=device)
                eval_correct = 0.0
                num_gt = len(valid_gt_label)

                if num_gt > 0:
                    cls_loss = loss_cls(valid_pred_label, valid_gt_label.long())
                    loss += 0.02 * cls_loss
                    pred = torch.max(valid_pred_label, 1)[1]
                    eval_correct = (pred == valid_gt_label).sum().item()

                if len(valid_gt_offset) > 0:
                    offset_loss = loss_offset(valid_pred_offset, valid_gt_offset)
                    loss += 0.6 * offset_loss

                if len(valid_landmark_offset) > 0:
                    landmark_loss = loss_landmark(valid_pred_landmark, valid_landmark_offset)
                    loss += 3.0 * landmark_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Update running statistics
                batch_size_actual = input_images.size(0)
                running_loss += loss.item() * batch_size_actual
                running_loss_cls += cls_loss.item() * batch_size_actual if num_gt > 0 else 0.0
                running_loss_offset += offset_loss.item() * batch_size_actual if len(valid_gt_offset) > 0 else 0.0
                running_loss_landmark += landmark_loss.item() * batch_size_actual if len(valid_landmark_offset) > 0 else 0.0
                running_correct += eval_correct
                running_gt += num_gt

        # Compute epoch statistics
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_loss_cls = running_loss_cls / dataset_sizes[phase] if running_loss_cls > 0 else 0.0
        epoch_loss_offset = running_loss_offset / dataset_sizes[phase] if running_loss_offset > 0 else 0.0
        epoch_loss_landmark = running_loss_landmark / dataset_sizes[phase] if running_loss_landmark > 0 else 0.0
        epoch_accuracy = running_correct / (running_gt + 1e-16)

        print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f} '
              f'Cls Loss: {epoch_loss_cls:.4f} Offset Loss: {epoch_loss_offset:.4f} '
              f'Landmark Loss: {epoch_loss_landmark:.4f}')

        # Save best model based on validation loss
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_accuracy = epoch_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

# Training summary
time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Loss: {best_loss:.4f}, Best val Accuracy: {best_accuracy:.4f}')

# Save the best model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Quant_Onet_weight.pth")
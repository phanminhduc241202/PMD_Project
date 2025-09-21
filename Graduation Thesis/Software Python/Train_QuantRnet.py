import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import cv2
import time
import copy
from collections import OrderedDict
from Quantization_MTCNN import QuantRNet 

# Dataset class for loading MTCNN data
class ListDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        annotation = self.img_files[index % len(self.img_files)].strip().split(' ')

        # Image
        img = cv2.imread(annotation[0])
        img = img[:,:,::-1]  # BGR to RGB
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = (img - 127.5) * 0.0078125  # Normalize
        input_img = torch.FloatTensor(img)

        # Label and targets
        label = int(annotation[1])
        bbox_target = np.zeros((4,))
        landmark = np.zeros((10,))

        if len(annotation[2:]) == 4:
            bbox_target = np.array(annotation[2:6]).astype(float)
        if len(annotation[2:]) == 14:
            bbox_target = np.array(annotation[2:6]).astype(float)
            landmark = np.array(annotation[6:]).astype(float)

        sample = {
            'input_img': input_img,
            'label': label,
            'bbox_target': bbox_target,
            'landmark': landmark
        }

        return sample

# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

def train_model(model, dataloaders, dataset_sizes, device, num_epochs=30):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_accuracy = 0.0

    optimizer = optim.Adam(model.parameters())
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()

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
            running_correct = 0.0
            running_gt = 0.0

            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                input_images = sample_batched['input_img'].to(device)
                gt_label = sample_batched['label'].to(device)
                gt_offset = sample_batched['bbox_target'].type(torch.FloatTensor).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred_offsets, pred_label = model(input_images)
                    # Trích xuất tensor từ IntQuantTensor (Brevitas)
                    pred_offsets = pred_offsets.value  # [batch_size, 4]
                    pred_label = pred_label.value  # [batch_size, 2]

                    # Classification loss
                    mask_cls = torch.ge(gt_label, 0)  # Positive (1) và negative (0)
                    valid_gt_label = gt_label[mask_cls]
                    valid_pred_label = pred_label[mask_cls]

                    # Offset loss
                    unmask = torch.eq(gt_label, 0)
                    mask_offset = torch.eq(unmask, 0)  # Positive (1) và part (-1)
                    valid_gt_offset = gt_offset[mask_offset]
                    valid_pred_offset = pred_offsets[mask_offset]

                    loss = torch.tensor(0.0).to(device)
                    cls_loss = torch.tensor(0.0).to(device)
                    offset_loss = torch.tensor(0.0).to(device)
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

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * input_images.size(0)
                    running_loss_cls += cls_loss.item() * input_images.size(0) if num_gt > 0 else 0
                    running_loss_offset += offset_loss.item() * input_images.size(0) if len(valid_gt_offset) > 0 else 0
                    running_correct += eval_correct
                    running_gt += num_gt

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_cls = running_loss_cls / dataset_sizes[phase]
            epoch_loss_offset = running_loss_offset / dataset_sizes[phase]
            epoch_accuracy = running_correct / (running_gt + 1e-16)

            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f} '
                  f'Cls Loss: {epoch_loss_cls:.4f} Offset Loss: {epoch_loss_offset:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}, Best val accuracy: {best_accuracy:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    # Data paths
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_24_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_24_val.txt"
    batch_size = 32

    # Data loaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True, num_workers=0)
    }
    dataset_sizes = {
        'train': len(ListDataset(train_path)),
        'val': len(ListDataset(val_path))
    }

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    model = QuantRNet(is_train=True).to(device)  # Thay QuantPNet bằng QuantRNet
    model.apply(weights_init)

    # Train
    model = train_model(model, dataloaders, dataset_sizes, device, num_epochs=30)

    # Save best model
    torch.save(model.state_dict(), "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Quant_Rnet_weight.pth")

if __name__ == '__main__':
    main()
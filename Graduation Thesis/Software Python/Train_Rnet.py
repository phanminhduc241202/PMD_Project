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

# Flatten layer for RNet
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)

# RNet model definition
class RNet(nn.Module):
    def __init__(self, is_train=False):
        super(RNet, self).__init__()
        self.is_train = is_train

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),
            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)  # Classification
        self.conv5_2 = nn.Linear(128, 4)  # Bounding box regression

    def forward(self, x):
        x = self.features(x)
        a = self.conv5_1(x)  # Classification scores
        b = self.conv5_2(x)  # Bounding box offsets
        if self.is_train is False:
            a = torch.nn.functional.softmax(a, dim=1)
        return b, a

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
                    pred_offsets = torch.squeeze(pred_offsets)
                    pred_label = torch.squeeze(pred_label)

                    # Classification loss
                    mask_cls = torch.ge(gt_label, 0)
                    valid_gt_label = gt_label[mask_cls]
                    valid_pred_label = pred_label[mask_cls]

                    # Offset loss
                    unmask = torch.eq(gt_label, 0)
                    mask_offset = torch.eq(unmask, 0)
                    valid_gt_offset = gt_offset[mask_offset]
                    valid_pred_offset = pred_offsets[mask_offset]

                    loss = torch.tensor(0.0).to(device)
                    cls_loss = 0.0
                    offset_loss = 0.0
                    eval_correct = 0.0
                    num_gt = len(valid_gt_label)

                    if len(valid_gt_label) != 0:
                        cls_loss = loss_cls(valid_pred_label, valid_gt_label)
                        loss += 0.02 * cls_loss
                        pred = torch.max(valid_pred_label, 1)[1]
                        eval_correct = (pred == valid_gt_label).sum().item()

                    if len(valid_gt_offset) != 0:
                        offset_loss = loss_offset(valid_pred_offset, valid_gt_offset)
                        loss += 0.6 * offset_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * input_images.size(0)
                    running_loss_cls += cls_loss * input_images.size(0) if isinstance(cls_loss, torch.Tensor) else 0
                    running_loss_offset += offset_loss * input_images.size(0) if isinstance(offset_loss, torch.Tensor) else 0
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
        'train': torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True)
    }
    dataset_sizes = {
        'train': len(ListDataset(train_path)),
        'val': len(ListDataset(val_path))
    }

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    model = RNet(is_train=True).to(device)
    model.apply(weights_init)

    # Train
    model = train_model(model, dataloaders, dataset_sizes, device, num_epochs=30)

    # Save best model
    torch.save(model.state_dict(), "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Rnet_weight.pth")

if __name__ == '__main__':
    main()
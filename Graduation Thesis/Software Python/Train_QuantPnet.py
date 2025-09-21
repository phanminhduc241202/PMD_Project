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
from Quantization_MTCNN import QuantPNet

# Dataset
class ListDataset(Dataset):
    def __init__(self, list_path):
        """
        Args:
            list_path (str): Path to annotation file.
        """
        with open(list_path, 'r') as file:
            self.img_files = [line.strip() for line in file if line.strip()]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        annotation = self.img_files[index].split()
        
        # Image
        img_path = annotation[0]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found")
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = (img - 127.5) * 0.0078125  # Normalize
        input_img = torch.FloatTensor(img)
        
        # Label and offsets
        label = int(annotation[1])
        bbox_target = np.zeros(4, dtype=np.float32)
        if label != 0:  # Positive (1) or part (-1)
            if len(annotation) == 6:
                bbox_target = np.array([float(x) for x in annotation[2:]], dtype=np.float32)
            else:
                print(f"Warning: Invalid annotation format at index {index}: {annotation}")
        
        sample = {
            'input_img': input_img,
            'label': label,
            'bbox_target': bbox_target
        }
        
        return sample

# Weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

def main():
    # Kaggle paths
    train_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_train.txt"
    val_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_store/imglist_anno_12_val.txt"
    batch_size = 32
    num_epochs = 15
    
    # Verify annotation files
    if not os.path.exists(train_path):
        print(f"Error: Training annotation file {train_path} not found.")
        return
    if not os.path.exists(val_path):
        print(f"Error: Validation annotation file {val_path} not found.")
        return
    
    # DataLoaders
    dataloaders = {
        'train': DataLoader(
            ListDataset(train_path),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        'val': DataLoader(
            ListDataset(val_path),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    }
    dataset_sizes = {
        'train': len(ListDataset(train_path)),
        'val': len(ListDataset(val_path))
    }
    print(f"Dataset sizes: train={dataset_sizes['train']}, val={dataset_sizes['val']}")
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = QuantPNet(is_train=True).to(device)
    model.apply(weights_init)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Loss functions
    loss_cls = nn.CrossEntropyLoss()
    loss_offset = nn.MSELoss()
    
    # Training loop
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    
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
                gt_offset = sample_batched['bbox_target'].to(device).float()
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    pred_offsets, pred_label = model(input_images)
                    
                    # Trích xuất tensor từ IntQuantTensor
                    pred_label = pred_label.value  # [batch_size, 2, h', w']
                    pred_offsets = pred_offsets.value  # [batch_size, 4, h', w']
                    
                    # Aggregate spatial dimensions
                    pred_label = pred_label.mean(dim=[2, 3])  # [batch_size, 2]
                    pred_offsets = pred_offsets.mean(dim=[2, 3])  # [batch_size, 4]
                    
                    # Classification loss
                    mask_cls = (gt_label >= 0)  # Positive (1) and negative (0)
                    valid_gt_label = gt_label[mask_cls]
                    valid_pred_label = pred_label[mask_cls]
                    
                    cls_loss = 0.0
                    eval_correct = 0.0
                    num_gt = len(valid_gt_label)
                    
                    if num_gt > 0:
                        cls_loss = loss_cls(valid_pred_label, valid_gt_label)
                        pred = torch.max(valid_pred_label, 1)[1]
                        eval_correct = (pred == valid_gt_label).sum().item()
                    
                    # Offset loss
                    mask_offset = (gt_label != 0)  # Positive (1) and part (-1)
                    valid_gt_offset = gt_offset[mask_offset]
                    valid_pred_offset = pred_offsets[mask_offset]
                    
                    offset_loss = 0.0
                    if len(valid_gt_offset) > 0:
                        offset_loss = loss_offset(valid_pred_offset, valid_gt_offset)
                    
                    # Combined loss
                    loss = 0.02 * cls_loss + 0.6 * offset_loss if (cls_loss > 0 or offset_loss > 0) else cls_loss
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * batch_size
                running_loss_cls += cls_loss.item() * batch_size if cls_loss > 0 else 0
                running_loss_offset += offset_loss.item() * batch_size if offset_loss > 0 else 0
                running_correct += eval_correct
                running_gt += num_gt
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_cls = running_loss_cls / dataset_sizes[phase]
            epoch_loss_offset = running_loss_offset / dataset_sizes[phase]
            epoch_accuracy = running_correct / (running_gt + 1e-16)
            
            print(f'{phase} Loss: {epoch_loss:.4f} accuracy: {epoch_accuracy:.4f} '
                  f'cls Loss: {epoch_loss_cls:.4f} offset Loss: {epoch_loss_offset:.4f}')
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best loss: {best_loss:.4f}')
    
    model.load_state_dict(best_model_wts)
    save_path = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/weights/Quan_Pnet_weight.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    main()
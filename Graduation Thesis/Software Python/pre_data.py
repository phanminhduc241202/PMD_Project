import os
import cv2
import time
from torch.utils.data import Dataset, DataLoader
import glob
import torch
import torchvision.transforms as transforms
import numpy as np

class WIDERFaceDataset(Dataset):
    def __init__(self, labels_dir, image_dir, transform=None, target_size=(640, 320)):
        """
        Initialize the WIDER FACE dataset for MTCNN training.
        
        Args:
            labels_dir (str): Directory containing .txt label files.
            image_dir (str): Directory containing WIDER FACE images.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Desired (height, width) for resized images.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.annotations = self._load_annotations(labels_dir)

    def _load_annotations(self, labels_dir):
        """
        Load annotations from individual .txt files in the labels directory.
        Expected format per .txt file: YOLO style
        One line per bounding box, e.g., class_id x_center y_center width height (normalized)
        """
        annotations = []
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for label_file in label_files:
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            possible_image_extensions = ['.jpg', '.jpeg', '.png']
            image_name = None
            for ext in possible_image_extensions:
                candidate = f"{base_name}{ext}"
                if os.path.exists(os.path.join(self.image_dir, candidate)):
                    image_name = candidate
                    break
            
            if image_name is None:
                print(f"Warning: No matching image found for {label_file}, skipping.")
                continue
            
            # Load image to get dimensions for denormalizing coordinates
            img_path = os.path.join(self.image_dir, image_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Cannot read image {img_path}, skipping.")
                continue
            orig_h, orig_w = image.shape[:2]
            
            bboxes = []
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            # YOLO format: class_id x_center y_center width height (normalized)
                            if len(parts) < 5:
                                print(f"Warning: Invalid box format in {label_file}, line: {line}")
                                continue
                            x_center, y_center, width, height = map(float, parts[1:5])
                            # Convert to x1 y1 x2 y2 (absolute pixel coordinates)
                            x1 = (x_center - width / 2) * orig_w
                            y1 = (y_center - height / 2) * orig_h
                            x2 = (x_center + width / 2) * orig_w
                            y2 = (y_center + height / 2) * orig_h
                            # Validate bounding box
                            if x2 > x1 and y2 > y1 and 0 <= x1 < orig_w and 0 <= y2 < orig_h:
                                bboxes.append([x1, y1, x2, y2])
                            else:
                                print(f"Warning: Invalid box dimensions in {label_file}, line: {line}")
                        except (ValueError, IndexError):
                            print(f"Warning: Invalid box format in {label_file}, line: {line}")
                            continue
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
                continue
            
            annotations.append({
                'image_name': image_name,
                'bboxes': bboxes
            })

        print(f"Loaded {len(annotations)} valid image-label pairs.")
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Returns:
            dict: Contains image and annotations.
        """
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation['image_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image and adjust bounding boxes
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.target_size[::-1])  # OpenCV uses (width, height)
        
        bboxes = []
        if annotation['bboxes']:
            for box in annotation['bboxes']:
                x1, y1, x2, y2 = box
                # Scale bounding boxes
                x1 = x1 * self.target_size[1] / orig_w
                y1 = y1 * self.target_size[0] / orig_h
                x2 = x2 * self.target_size[1] / orig_w
                y2 = y2 * self.target_size[0] / orig_h
                bboxes.append([x1, y1, x2, y2])
        
        sample = {
            'image': image,
            'bboxes': np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0, 4), dtype=np.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def custom_collate_fn(batch):
    """
    Custom collation to handle variable number of bounding boxes.
    
    Args:
        batch: List of samples from the dataset.
        
    Returns:
        dict: Batched images and bounding boxes.
    """
    images = []
    bboxes = []
    
    for sample in batch:
        images.append(torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0)
        bboxes.append(torch.tensor(sample['bboxes'], dtype=torch.float32))
    
    return {
        'image': torch.stack(images),
        'bboxes': bboxes  # List of tensors, as number of boxes varies
    }

def convert_txt_annotations(labels_dir, output_txt, image_dir):
    """
    Process individual .txt label files and combine into a single annotation file.
    
    Args:
        labels_dir (str): Directory containing .txt label files.
        output_txt (str): Path to save processed annotations.
        image_dir (str): Directory containing images.
    """
    line_count = 0
    box_count = 0
    print('Starting annotation processing...')
    start_time = time.time()

    with open(output_txt, 'w') as f:
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        for label_file in label_files:
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            possible_image_extensions = ['.jpg', '.jpeg', '.png']
            image_name = None
            for ext in possible_image_extensions:
                candidate = f"{base_name}{ext}"
                img_path = os.path.join(image_dir, candidate)
                if os.path.exists(img_path):
                    image_name = candidate
                    break
            
            if image_name is None:
                print(f"Warning: No matching image for {label_file}, skipping.")
                continue
            
            # Load image to get dimensions
            img_path = os.path.join(image_dir, image_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Cannot read image {img_path}, skipping.")
                continue
            orig_h, orig_w = image.shape[:2]
            
            bboxes = []
            try:
                with open(label_file, 'r') as infile:
                    lines = infile.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            if len(parts) < 5:
                                print(f"Warning: Invalid box format in {label_file}, line: {line}")
                                continue
                            x_center, y_center, width, height = map(float, parts[1:5])
                            # Convert to x1 y1 x2 y2 (absolute pixel coordinates)
                            x1 = (x_center - width / 2) * orig_w
                            y1 = (y_center - height / 2) * orig_h
                            x2 = (x_center + width / 2) * orig_w
                            y2 = (y_center + height / 2) * orig_h
                            if x2 > x1 and y2 > y1 and 0 <= x1 < orig_w and 0 <= y2 < orig_h:
                                bboxes.append(f"{x1} {y1} {x2} {y2}")
                                box_count += 1
                            else:
                                print(f"Warning: Invalid box dimensions in {label_file}, line: {line}")
                        except (ValueError, IndexError):
                            print(f"Warning: Invalid box format in {label_file}, line: {line}")
                            continue
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
                continue
            
            line_count += 1
            if bboxes:
                line_str = f"{image_name} {' '.join(bboxes)}\n"
            else:
                line_str = f"{image_name}\n"
            f.write(line_str)

    elapsed_time = time.time() - start_time
    print('Finished processing annotations.')
    print(f'Time spent: {elapsed_time:.2f} seconds')
    print(f'Total images: {line_count}')
    print(f'Total boxes (faces): {box_count}')

def main():
    # Kaggle-specific paths
    image_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/valid/images"
    labels_dir = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/dataset/valid/labels"
    output_annotation_file = "C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output/anno_valid_processed.txt"

    # Ensure output directory exists
    os.makedirs('C:/Minh_Duc/MD_Personal/LVTN/Training_MTCNN/output', exist_ok=True)

    # Process annotations
    convert_txt_annotations(labels_dir, output_annotation_file, image_dir)

    # Define basic transform
    transform = transforms.Compose([
        # Additional transforms can be added here
    ])

    # Initialize dataset
    dataset = WIDERFaceDataset(
        labels_dir=labels_dir,
        image_dir=image_dir,
        transform=transform,
        target_size=(640, 320)  # Adjust as needed for MTCNN
    )

    # Example DataLoader for MTCNN training
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    # Test dataset
    print(f"Dataset size: {len(dataset)}")
    try:
        for batch in dataloader:
            print("Sample batch loaded:", {
                'image_shape': batch['image'].shape,
                'bboxes_lengths': [len(b) for b in batch['bboxes']]
            })
            break
    except Exception as e:
        print(f"Error in DataLoader: {e}")

if __name__ == "__main__":
    main()


   
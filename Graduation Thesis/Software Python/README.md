# Development and Optimization of a Custom Multi-Task Cascaded CNN for Efficient Edge Face Detection

A lightweight and optimized face detection system based on MTCNN architecture, designed for real-time deployment on resource-constrained edge devices and FPGAs.

## 📋 Project Overview

This project develops a customized Multi-Task Cascaded Convolutional Neural Network (MTCNN) optimized for efficient face detection on edge computing platforms. The system addresses the challenge of deploying high-accuracy face detection models on hardware-constrained devices while maintaining real-time performance.

## 🎯 Key Objectives

- **Architectural Optimization**: Redesign MTCNN with reduced parameters while maintaining accuracy
- **Edge Deployment**: Enable real-time face detection on FPGA and ARM-based systems
- **Model Compression**: Implement quantization-aware training and advanced compression techniques
- **Hardware Efficiency**: Optimize for deployment on Kria KV260 FPGA platform

## 🏗️ System Architecture

The system implements a three-stage cascaded CNN architecture:

```
Input Image → [P-Net] → [R-Net] → [O-Net] → Face Detection + Landmarks
               ↓         ↓         ↓
           Proposals   Refinement  Final Output
```

### Network Components

- **P-Net (Proposal Network)**: Generates initial face candidates from image pyramid
- **R-Net (Refinement Network)**: Filters false positives and refines bounding boxes  
- **O-Net (Output Network)**: Produces final detection results and facial landmarks

## 🚀 Key Features

### Custom Optimizations
- **Depthwise Separable Convolutions**: Replace standard convolutions to reduce computational cost
- **Squeeze-and-Excitation (SE) Blocks**: Maintain accuracy while reducing model complexity
- **Quantization-Aware Training (QAT)**: Enable INT8 inference for faster edge deployment
- **Image Pyramid Optimization**: Adjusted scaling factor (0.707) for improved efficiency

### Performance Achievements
- **60% Parameter Reduction**: From 495,850 to 197,202 parameters
- **2× Speed Improvement**: Compared to original MTCNN
- **96.3% Average Precision**: On WIDER-FACE dataset (vs 97.8% original)
- **Real-time Performance**: 30 FPS on Kria KV260 FPGA platform

## 🛠️ Technical Implementation

### Model Architecture Details

#### Customized P-Net
- Input: 12×12×3 images
- Reduced from 10→8 filters in first layer
- Integrated SE blocks for feature enhancement
- Parameters: 1,846 (vs 6,632 original)

#### Customized R-Net  
- Input: 24×24×3 images
- Depthwise separable convolutions
- Reduced filter counts while maintaining accuracy
- Parameters: 79,174 (vs 100,178 original)

#### Customized O-Net
- Input: 48×48×3 images
- Enhanced feature extraction with SE blocks
- Optimized fully connected layers
- Parameters: 116,182 (vs 389,040 original)

### Quantization Strategy
- **Post-Training Quantization (PTQ)**: Static and dynamic quantization
- **INT8 Deployment**: Reduced memory footprint and faster inference
- **FBGEMM Backend**: Optimized for x86 CPU deployment
- **QNNPACK Support**: ARM-based device compatibility

## 📊 Performance Results

### Accuracy Comparison
| Network | Original Accuracy | Optimized Accuracy | Parameter Reduction |
|---------|------------------|-------------------|-------------------|
| P-Net   | 98.7%           | 95.17%            | 72.2%            |
| R-Net   | 99.1%           | 91.7%             | 21.0%            |
| O-Net   | 99.4%           | 94.8%             | 70.1%            |

### Hardware Performance
| Platform | FPS | Power Consumption | Model Size |
|----------|-----|------------------|------------|
| Kria KV260 FPGA | 30 | ~1.1W | 290 KB |
| ARM Cortex-A53 | 20-25 | ~2.4W | 290 KB |
| Intel i7 CPU | 50+ | N/A | 290 KB |

### Comparison with Other Methods
| Method | Accuracy | FPS (CPU) | Model Size | Real-time Edge |
|--------|----------|-----------|------------|---------------|
| Haar Cascade | ~72% | 25 | Small | ✓ |
| DLIB HOG | ~82% | ~20 | Medium | ✓ |
| Original MTCNN | 98.8% | 15 | 6.7 MB | ✗ |
| **Our Method** | **96.3%** | **30+** | **290 KB** | **✓** |

## 💻 Installation & Usage

### Prerequisites
```bash
pip install torch torchvision opencv-python numpy matplotlib
pip install brevitas  # For quantization
```

### Quick Start
```python
import cv2
from mtcnn_optimized import OptimizedMTCNN

# Initialize detector
detector = OptimizedMTCNN()

# Load and detect faces
image = cv2.imread('path/to/image.jpg')
faces, landmarks = detector.detect_faces(image)

# Visualize results
for face in faces:
    x, y, w, h = face['box']
    confidence = face['confidence']
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Face Detection', image)
cv2.waitKey(0)
```

### Training Custom Model
```python
from training.train_mtcnn import MTCNNTrainer

# Initialize trainer
trainer = MTCNNTrainer(config_path='config/train_config.yaml')

# Train individual networks
trainer.train_pnet(epochs=100)
trainer.train_rnet(epochs=80) 
trainer.train_onet(epochs=60)

# Apply quantization
trainer.quantize_models(method='qat')
```

## 📁 Project Structure

```
├── models/
│   ├── pnet.py              # P-Net implementation
│   ├── rnet.py              # R-Net implementation
│   ├── onet.py              # O-Net implementation
│   └── mtcnn_optimized.py   # Main MTCNN class
├── training/
│   ├── train_mtcnn.py       # Training pipeline
│   ├── data_preparation.py   # Dataset handling
│   └── quantization.py      # Model quantization
├── utils/
│   ├── image_utils.py       # Image processing utilities
│   ├── nms.py              # Non-maximum suppression
│   └── visualization.py     # Result visualization
├── config/
│   └── train_config.yaml   # Training configuration
├── deployment/
│   ├── fpga_deploy.py      # FPGA deployment scripts
│   └── edge_inference.py   # Edge device inference
└── examples/
    ├── face_detection_demo.py
    └── benchmark_performance.py
```

## 🔧 Advanced Configuration

### Custom Architecture
```python
# Modify network architecture
config = {
    'pnet': {
        'filters': [8, 16, 32],
        'use_se_block': True,
        'se_reduction': 4
    },
    'rnet': {
        'filters': [16, 32, 64],
        'use_depthwise': True
    },
    'onet': {
        'filters': [16, 32, 48, 64],
        'fc_units': [128, 256]
    }
}
```

### Quantization Settings
```python
quantization_config = {
    'method': 'qat',  # 'ptq' or 'qat'
    'bits': 8,
    'calibration_samples': 500,
    'backend': 'fbgemm'  # 'qnnpack' for ARM
}
```

## 🎯 Applications

- **Security Systems**: Real-time surveillance and access control
- **Mobile Devices**: Face detection for photography and AR
- **Automotive**: Driver monitoring and attention detection
- **IoT Devices**: Smart home and retail analytics
- **Healthcare**: Patient monitoring and elderly care

## 📈 Benchmarking

### Robustness Testing
| Condition | Original MTCNN | Our Method | Improvement |
|-----------|---------------|------------|-------------|
| Partial Occlusion | 91% | 89% | Minimal loss |
| Varying Illumination | 95% | 93% | Minimal loss |
| Extreme Poses | 87% | 84% | Acceptable trade-off |
| Low Resolution | 78% | 75% | Maintained performance |

### Edge Device Deployment
- **Kria KV260**: 30 FPS, 1.1W power consumption
- **Raspberry Pi 4**: 15 FPS, efficient deployment
- **Jetson Nano**: 25 FPS, GPU acceleration support

## 🔮 Future Enhancements

### Planned Features
- **WiFi Connectivity**: Remote monitoring capabilities
- **Mobile App Integration**: Real-time face detection on smartphones
- **Advanced Timer Functions**: Scheduled detection operations
- **Voice Control**: Hands-free operation
- **Energy Monitoring**: Power consumption optimization

### Technical Improvements
- **Enhanced Quantization**: Mixed-precision deployment
- **Model Distillation**: Further size reduction
- **Dynamic Scaling**: Adaptive performance based on hardware
- **Multi-face Tracking**: Temporal consistency across frames

## 📚 Academic Context

This project was developed as part of the Computer Engineering Technology program at Ho Chi Minh City University of Technology and Education, demonstrating:

- **Deep Learning Optimization**: Advanced model compression techniques
- **Edge Computing**: Practical deployment considerations
- **Hardware-Software Co-design**: FPGA acceleration strategies
- **Computer Vision**: Real-world application development

## 🏆 Key Contributions

1. **Novel Architecture**: Custom MTCNN with SE blocks and depthwise separable convolutions
2. **Quantization Strategy**: Comprehensive QAT implementation for edge deployment
3. **Hardware Optimization**: Efficient FPGA deployment achieving real-time performance
4. **Benchmark Results**: Extensive evaluation on standard face detection datasets

## 👥 Team

- **Students**: Phan Minh Duc (21119303), Nguyen Le Ngoc Tram (21119322)
- **Supervisor**: Dr. Pham Van Khoa
- **Institution**: Ho Chi Minh City University of Technology and Education

## 📄 License

This project is developed for educational and research purposes. Commercial use requires proper attribution and licensing agreements.

## 🙏 Acknowledgments

We thank the Faculty of International Education and Dr. Pham Van Khoa for guidance and support. Special appreciation to colleagues who provided valuable feedback throughout the development process.

---

*For detailed technical documentation, training procedures, and deployment guides, please refer to the complete thesis document and source code in this repository.*

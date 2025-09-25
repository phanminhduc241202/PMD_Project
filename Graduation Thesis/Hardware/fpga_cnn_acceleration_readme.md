# Convolutional Neural Network Acceleration for Face Detection on FPGA

A high-performance FPGA-based hardware accelerator for CNN inference, specifically optimized for MTCNN face detection using High-Level Synthesis (HLS) techniques.

## Project Overview

This project implements a comprehensive FPGA acceleration solution for Convolutional Neural Networks, focusing on the Multi-Task Cascaded Convolutional Network (MTCNN) architecture for real-time face detection. The system leverages the parallel processing capabilities of FPGAs to achieve significant performance improvements over traditional CPU implementations.

## Key Objectives

- **CNN Hardware Acceleration**: Implement efficient CNN inference on FPGA hardware
- **Real-time Performance**: Achieve low-latency face detection for time-critical applications
- **Power Efficiency**: Optimize power consumption for edge deployment
- **HLS Optimization**: Utilize advanced HLS techniques for hardware design optimization

## System Architecture

### Hardware Platform
- **Target Device**: AMD Xilinx Kria KV260 Vision AI Starter Kit
- **SoC**: Zynq UltraScale+ MPSoC
- **Architecture**: Processing System (PS) + Programmable Logic (PL)

### Processing Components
```
┌─────────────────┐    ┌──────────────────┐
│   ARM Cortex    │    │  FPGA Fabric     │
│   A53/R5F       │◄──►│  (Custom Logic)  │
│  (Host Control) │    │  (CNN Kernels)   │
└─────────────────┘    └──────────────────┘
         │                        │
         └────────────────────────┘
              AXI Interface
```

## Technical Specifications

### Hardware Resources (Kria KV260)
| Resource Type | Available | Utilization | Usage |
|---------------|-----------|-------------|-------|
| LUT | 117,120 | 77.95% | 91,292 |
| Flip-Flop | 234,240 | 49.31% | 115,497 |
| BRAM | 144 | 87.85% | 126.5 |
| DSP | 1,248 | 40.22% | 502 |
| URAM | 64 | 25.00% | 16 |

### Performance Results
| Network | Intel i7-14650HX (2.2GHz) | FPGA (100MHz) | Speedup |
|---------|---------------------------|---------------|---------|
| P-Net | 1.2ms | 0.035ms | **34×** |
| R-Net | 1.5ms | 0.810ms | **1.9×** |
| O-Net | 2.6ms | 3.525ms | Comparable |

## Key Features

### HLS Optimization Techniques
- **Pipelining**: Overlap computation stages for increased throughput
- **Loop Unrolling**: Parallelize loop iterations for faster execution
- **Dataflow Architecture**: Enable concurrent processing across network layers
- **On-chip Buffering**: Reduce external memory access overhead

### Custom Optimizations
- **Depthwise Separable Convolutions**: Reduce computational complexity
- **Squeeze-and-Excitation Blocks**: Maintain accuracy with fewer parameters
- **Stream Buffers**: Enable efficient data flow between processing stages
- **Memory Access Optimization**: Minimize loop-carried dependencies

### Network Implementations
1. **P-Net (Proposal Network)**
   - Input: 12×12×3
   - Fully convolutional architecture
   - Resource-optimized for high-throughput candidate generation

2. **R-Net (Refinement Network)**  
   - Input: 24×24×3
   - Efficient false positive filtering
   - Balanced performance-accuracy trade-off

3. **O-Net (Output Network)**
   - Input: 48×48×3
   - High-precision final detection
   - Integrated landmark localization

## Implementation Details

### Development Flow
```mermaid
graph LR
    A[C/C++ Code] --> B[Vitis HLS]
    B --> C[RTL Generation]
    C --> D[Vivado Synthesis]
    D --> E[Implementation]
    E --> F[Bitstream]
    F --> G[FPGA Deployment]
```

### Layer Optimization Strategies

#### Convolutional Layer
```cpp
// On-chip buffering for convolution
void conv_layer_optimized(
    float input[IN_CH][IN_H][IN_W],
    float output[OUT_CH][OUT_H][OUT_W],
    float weights[OUT_CH][IN_CH][K_H][K_W]
) {
    #pragma HLS DATAFLOW
    
    // Processing loops with optimization pragmas
    CONV_LOOP_OUT: for(int n = 0; n < OUT_CH; n++) {
        CONV_LOOP_H: for(int h = 0; h < OUT_H; h++) {
            #pragma HLS PIPELINE II=1
            CONV_LOOP_W: for(int w = 0; w < OUT_W; w++) {
                #pragma HLS UNROLL
                // Optimized convolution computation
            }
        }
    }
}
```

#### On-chip Buffer Strategy
- **Buffer Size**: Kernel dimensions (KX × KY)
- **Memory Type**: BRAM for fast access
- **Access Pattern**: Eliminates loop-carried dependencies
- **Parallelization**: Enables efficient pipelining/unrolling

### Resource Utilization Breakdown

#### P-Net Kernel
- **LUT Usage**: 24,312 (20.76%)
- **BRAM Usage**: 5 (3.5%)
- **DSP Usage**: 162 (12.98%)
- **Optimization**: Highest acceleration ratio achieved

#### R-Net Kernel
- **LUT Usage**: 28,959 (24.73%)
- **BRAM Usage**: 35.5 (20.83%)
- **DSP Usage**: 191 (15.3%)
- **Feature**: Balanced resource-performance trade-off

#### O-Net Kernel
- **LUT Usage**: 25,608 (21.86%)
- **BRAM Usage**: 86 (59.72%)
- **DSP Usage**: 149 (11.94%)
- **Challenge**: Largest network requiring careful optimization

## Power Analysis

### System Power Consumption
- **Total Power**: 3.544W
- **Processing System**: 2.419W
- **Hardware Accelerators**: 1.125W
- **Efficiency**: Significant improvement over CPU-only solutions

### Power Breakdown
```
┌─────────────────────────────────────┐
│ Dynamic Power: 2.670W              │
│ ├─PS: 2.419W                       │
│ └─PL: 0.251W                       │
│                                     │
│ Static Power: 0.874W               │
│ ├─PS: 0.000W                       │
│ └─PL: 0.874W                       │
│                                     │
│ Total: 3.544W                      │
└─────────────────────────────────────┘
```

## Software Implementation

### Development Environment
```bash
# Required tools
- Xilinx Vitis HLS 2023.1
- Xilinx Vivado 2023.1
- PYNQ Framework
- Python 3.8+
- OpenCV, NumPy
```

### Deployment Pipeline
```python
# Load bitstream and control accelerators
from pynq import Overlay
import numpy as np

# Load FPGA design
overlay = Overlay('mtcnn_accelerator.bit')

# Allocate memory buffers
input_buffer = overlay.allocate(shape=(12,12,3), dtype=np.float32)
output_buffer = overlay.allocate(shape=(2,), dtype=np.float32)

# Execute P-Net kernel
overlay.pnet_kernel.call(input_buffer, output_buffer)
```

### Integration Example
```cpp
// Host application for MTCNN acceleration
void mtcnn_accelerated_inference(cv::Mat& image) {
    // Image pyramid generation
    std::vector<cv::Mat> pyramid = generate_pyramid(image);
    
    // P-Net processing
    for(auto& img : pyramid) {
        auto proposals = pnet_accelerator(img);
        // NMS and refinement
        proposals = apply_nms(proposals);
    }
    
    // R-Net and O-Net processing
    auto refined = rnet_accelerator(proposals);
    auto final = onet_accelerator(refined);
}
```

## Performance Benchmarks

### Latency Comparison
| Implementation | P-Net | R-Net | O-Net | Total |
|----------------|-------|-------|-------|-------|
| CPU (i7-14650HX) | 1.2ms | 1.5ms | 2.6ms | 5.3ms |
| **FPGA (Optimized)** | **0.035ms** | **0.810ms** | **3.525ms** | **4.37ms** |
| Speedup | 34.3× | 1.9× | 0.7× | 1.2× |

### Throughput Analysis
- **Clock Frequency**: 100MHz (vs 2.2GHz CPU)
- **Parallel Processing**: Up to 32 concurrent operations
- **Memory Bandwidth**: Optimized on-chip access patterns
- **Pipeline Efficiency**: II=1 achieved for critical loops

## Advanced Optimization Techniques

### 1. Dataflow Architecture
```cpp
#pragma HLS DATAFLOW
void mtcnn_pipeline(
    stream<data_t>& input,
    stream<data_t>& output
) {
    static stream<data_t> s1, s2, s3;
    
    conv_layer1(input, s1);
    pooling_layer1(s1, s2);
    conv_layer2(s2, s3);
    output_layer(s3, output);
}
```

### 2. Memory Optimization
- **Stream Interfaces**: FIFO-based data flow
- **Burst Transfers**: Efficient AXI memory access
- **Buffer Management**: Minimize external memory traffic
- **Data Reuse**: Exploit spatial/temporal locality

### 3. Precision Optimization
- **Fixed-Point Arithmetic**: Reduced resource usage
- **Bit-width Analysis**: Minimize precision while maintaining accuracy
- **Quantization**: Post-training quantization support

## Project Structure

```
├── src/
│   ├── layers/
│   │   ├── conv_layer.cpp          # Convolutional layer implementation
│   │   ├── pooling_layer.cpp       # Pooling operations
│   │   ├── fc_layer.cpp           # Fully connected layers
│   │   └── activation.cpp         # Activation functions
│   ├── networks/
│   │   ├── pnet_kernel.cpp        # P-Net FPGA kernel
│   │   ├── rnet_kernel.cpp        # R-Net FPGA kernel
│   │   └── onet_kernel.cpp        # O-Net FPGA kernel
│   └── utils/
│       ├── data_types.hpp         # Custom data types
│       └── optimization.hpp       # HLS optimization pragmas
├── testbench/
│   ├── tb_pnet.cpp               # P-Net testbench
│   ├── tb_rnet.cpp               # R-Net testbench
│   └── tb_onet.cpp               # O-Net testbench
├── vivado/
│   ├── block_design.tcl          # Vivado block design
│   └── constraints.xdc           # Timing constraints
├── host/
│   ├── mtcnn_host.cpp           # Host application
│   └── image_processing.cpp      # Image preprocessing
├── scripts/
│   ├── build_hls.tcl            # HLS build script
│   ├── build_vivado.tcl         # Vivado build script
│   └── deploy.py                # Deployment utilities
└── docs/
    ├── architecture.md          # Detailed architecture
    ├── optimization.md          # Optimization techniques
    └── benchmarks.md           # Performance results
```

## Getting Started

### 1. Environment Setup
```bash
# Source Xilinx tools
source /opt/Xilinx/Vitis/2023.1/settings64.sh
source /opt/Xilinx/Vivado/2023.1/settings64.sh

# Install PYNQ
pip install pynq

# Clone repository
git clone <repository-url>
cd fpga-cnn-acceleration
```

### 2. Build Hardware
```bash
# HLS synthesis
vitis_hls -f scripts/build_hls.tcl

# Vivado implementation
vivado -mode batch -source scripts/build_vivado.tcl

# Generate bitstream
# Output: mtcnn_accelerator.bit
```

### 3. Deploy and Test
```bash
# Copy to Kria KV260
scp *.bit *.hwh root@kria-board:/home/root/

# Run host application
python host/test_mtcnn.py --image test_image.jpg
```

## Applications

### Target Use Cases
- **Automotive**: Driver monitoring systems
- **Security**: Real-time surveillance
- **Mobile**: Edge AI applications
- **Healthcare**: Patient monitoring
- **Industrial**: Quality control systems

### Deployment Scenarios
- **Edge Computing**: Low-latency local processing
- **IoT Devices**: Power-efficient operation  
- **Embedded Systems**: Resource-constrained environments
- **Real-time Systems**: Deterministic performance requirements

## Future Enhancements

### Hardware Optimizations
- **Dynamic Partial Reconfiguration**: Runtime architecture adaptation
- **Mixed-Precision**: INT8/INT16 quantization support
- **Memory Hierarchy**: Advanced caching strategies
- **Multi-core Processing**: Parallel CNN execution

### Software Improvements
- **Model Compression**: Pruning and distillation
- **Runtime Optimization**: Dynamic load balancing
- **Performance Monitoring**: Real-time profiling
- **Error Handling**: Robust fault tolerance

### Integration Features
- **Video Processing**: Real-time video stream analysis
- **Multi-model Support**: Flexible CNN architecture support
- **Cloud Connectivity**: Hybrid edge-cloud processing
- **API Development**: Standardized acceleration interfaces

## Academic Context

This research was conducted at Ho Chi Minh City University of Technology and Education, contributing to:

- **FPGA-based AI Acceleration**: Advanced hardware design techniques
- **High-Level Synthesis**: Efficient hardware-software co-design
- **Computer Vision**: Real-time image processing systems
- **Edge Computing**: Resource-optimized AI deployment

## Key Contributions

1. **Efficient HLS Implementation**: Optimized CNN layers for FPGA deployment
2. **On-chip Buffering Strategy**: Novel approach to eliminate memory bottlenecks
3. **Comprehensive Benchmarking**: Detailed performance analysis across multiple metrics
4. **Complete System Integration**: End-to-end deployment on Kria KV260 platform

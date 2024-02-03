# Advancing-the-Architecture-of-Restoring-Extremely-Dark-Images-in-Real-Time
# Image Restoration Project - README

##Youtube link:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/BMiy5SJKojA/0.jpg)](https://www.youtube.com/watch?v=BMiy5SJKojA)
(https://youtu.be/BMiy5SJKojA)

## Introduction
Welcome to the Image Restoration Project! This project focuses on advancing the architecture for restoring extremely dark images in real time. The goal is to balance restoration quality, computational efficiency, and speed. We build upon Lamba and Mitra's foundational work, aiming to refine their deep learning model and push the boundaries of low-light image enhancement.

## Objective
The primary objective of this project is to enhance the existing deep learning model, seeking marked improvements in image quality, processing speed, and computational efficiency. The aim is to set a new standard in the field of computational photography, particularly in low-light image enhancement scenarios.

## Methodology
The project involves two primary stages: Model Development and Comparative Analysis.

### Model Development
1. **MO Architecture (Original Architecture):**
   - Core Design: U-net style architecture with downsampling, Residual Dense Blocks (RDBs), and upsampling.
   - Feature Extraction: Downsample using a downshuffle method, RDBs capture complex features, and features from different levels are concatenated and upsampled.
   - Overall Structure: Multi-scale approach processing images at different resolutions (32x, 8x, 2x).

2. **M1 and M2 Architecture (Modified Architecture):**
   - Modifications: Sequence and quantity of downsampling layers altered for better feature extraction.
   - M1: Introduces two additional layers (8x with RDB and 16x) to enhance feature extraction.
   - M2: Maintains the same number of downsampling layers as M1 but introduces a novel approach in combining these layers using additional RDBs.

### Comparative Analysis
Comprehensive analysis conducted, comparing MO, M1, and M2 models on diverse datasets. Evaluation focused on image quality, processing speed, and generalization capabilities.

## Architecture Implementation
Detailed explanations of the original (MO) and modified (M1, M2) architectures are provided, outlining the key components such as downsampling, RDBs, feature concatenation, and upsampling.

## Experiments
### Training Phase
- Due to computational demands, each training session extended beyond 72 hours.
- Models trained for 10,000 epochs for experimental purposes.
- Training loss and accuracy metrics observed during the training process.

### Testing Phase
- Extensive testing on the validation dataset, evaluating PSNR and visual quality.
- Evaluation extended to formats not present in the training set.
- Model's latency measured for processing performance in terms of time efficiency.

## Results
- Training loss and PSNR values provided for MO, M1, and M2 models.
- Visual comparison of restored images demonstrates improvements in image quality for M1 and M2.
- MO exhibits a speed advantage over M1 and M2, but trade-offs between speed and image quality must be considered.

## Conclusion
In conclusion, model M1 outperforms the original architecture in terms of performance while maintaining comparable time complexity. The results suggest the potential for further enhancement with ongoing research and refinement.

Feel free to explore the detailed documentation and experiment results for a deeper understanding of the project.

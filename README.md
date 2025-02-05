# Awesome-Depth-Estimation
A curated list of papers and resources focused on Depth Estimation. 

## Table of Contents

1. [Selected Papers on Depth Estimation](#selected-papers-on-depth-estimation)
2. [Useful Datasets](#useful-datasets)
3. [Evaluation Metrics](#evaluation-metrics)

---

## Selected Papers on Depth Estimation

### **UniDepth: Universal Monocular Metric Depth Estimation**  

Authors: Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu  

Published: CVPR 2024

**[Paper](https://arxiv.org/pdf/2403.18913)** | **[Project](https://lpiccinelli-eth.github.io/pub/unidepth/)** | **[Code](https://github.com/lpiccinelli-eth/UniDepth)**

***Keywords***: *Monocular Depth Estimation*, *Transformer based*, *Images Depth*

<details>
  <summary>Click to view Abstract</summary>

  Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepth, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE methods, UniDepth directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepth implements a self-promptable camera module predicting dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. Thorough evaluations on ten datasets in a zero-shot regime consistently demonstrate the superior performance of UniDepth, even when compared with methods directly trained on the testing domains.

</details>

---

## Useful Datasets

Here is a list of useful datasets for depth estimation:

### **NYU Depth Dataset V2**  
Authors: Nathan Silberman, Pushmeet Kohli, Derek Hoiem, Rob Fergus  
Published: ECCV 2012

**[Paper](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf)** | **[Project](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)**

---

### **ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes**  
Authors: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner

**[Paper](http://www.scan-net.org/)** | **[Project](https://github.com/ScanNet/ScanNet)**

---

### **SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels**  
Authors: Jianxiong Xiao, Andrew Owens, Antonio Torralba  
Published: Proceedings of 14th IEEE International Conference on Computer Vision (ICCV2013)

**[Paper](https://vision.princeton.edu/projects/2013/SUN3D/paper.pdf)** | **[Project](https://sun3d.cs.princeton.edu/)**

---





---

## Evaluation Metrics

To evaluate depth estimation models, the following metrics are commonly used:

1. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and ground truth depth values.
2. **Root Mean Squared Error (RMSE)**: Measures the square root of the mean squared differences between predicted and ground truth depth values.
3. **Threshold Accuracy**: Measures the percentage of predictions that are within a certain threshold of the ground truth.

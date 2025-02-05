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

**[Paper](http://www.scan-net.org/)** | **[Project](https://github.com/ScanNet/ScanNet)** | **[Code](https://github.com/ScanNet/ScanNet)**

---

### **SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels**  
Authors: Jianxiong Xiao, Andrew Owens, Antonio Torralba  
Published: Proceedings of 14th IEEE International Conference on Computer Vision (ICCV2013)

**[Paper](https://vision.princeton.edu/projects/2013/SUN3D/paper.pdf)** | **[Project](https://sun3d.cs.princeton.edu/)**

---

### **KITTI**  
Authors: Andreas Geiger, Philip Lenz, Christoph Stiller, Raquel Urtasun  

**[Project](https://www.cvlibs.net/datasets/kitti/)**

---

### **DDAD-Dense Depth for Autonomous Driving**  
Authors: Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, Adrien Gaidon  
Published: CVPR 2020  

**[Project](https://github.com/TRI-ML/DDAD)**

---

### **DIODE: A Dense Indoor and Outdoor DEpth Dataset**  
Authors: Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, Gregory Shakhnarovich  

**[Project](https://diode-dataset.org/)** | **[Paper](https://arxiv.org/pdf/1908.00463)** | **[Code](https://github.com/diode-dataset/diode-devkit)**

---

### **Hypersim**  
Authors: Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, Joshua M. Susskind  
Published: ICCV 2021

**[Project](https://github.com/apple/ml-hypersim)**



---

## Evaluation Metrics

To evaluate depth estimation models, the following metrics are commonly used:

1. **AbsRel (Absolute Relative Error)**:
   - **Definition**: Measures the average relative error between predicted depth and ground truth depth.
   - **Formula**:
     $$
     \text{AbsRel} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\hat{d}_i - d_i|}{d_i}
     $$  
     Where \( \hat{d}_i \) is the predicted depth, \( d_i \) is the ground truth depth, and \( N \) is the number of pixels or points.

2. **δ₁ (Delta 1)**:
   - **Definition**: Measures the percentage of predictions where the predicted depth is within a factor of 1.0 of the ground truth depth.
   - **Formula**:
     $$
     \delta_1 = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left(\frac{\hat{d}_i}{d_i} < 1.0\right)
     $$  
     Where \( \hat{d}_i \) is the predicted depth, \( d_i \) is the ground truth depth, and \( \mathbf{1} \) is an indicator function that equals 1 if the condition is true, otherwise 0.

3. **RMS (Root Mean Squared Error)**:
   - **Definition**: Measures the square root of the average squared differences between predicted and ground truth depths.
   - **Formula**:
     $$
     RMS = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{d}_i - d_i)^2}
     $$  
     Where \( \hat{d}_i \) is the predicted depth, \( d_i \) is the ground truth depth, and \( N \) is the number of pixels or points.

4. **RMSlog (Logarithmic Root Mean Squared Error)**:
   - **Definition**: Similar to RMS, but applied to the logarithms of the depth values to focus on the relative differences.
   - **Formula**:
     $$
     RMS_{\log} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\log(\hat{d}_i) - \log(d_i)\right)^2}
     $$  
     Where \( \log \) represents the logarithmic transformation.

5. **CD (Chernoff Distance)**:
   - **Definition**: Measures the difference between predicted and ground truth depth distributions.
   - **Formula**: Typically computed using statistical distance measures such as KL divergence or other distribution-based methods.

6. **SIlog (Scaled Logarithmic Error)**:
   - **Definition**: Evaluates the error in the logarithmic scale of depth values, scaled to the range of depths.
   - **Formula**:
     $$
     SI_{\log} = \frac{1}{N} \sum_{i=1}^{N} \left|\log(\hat{d}_i) - \log(d_i)\right|
     $$  
     Where \( \hat{d}_i \) is the predicted depth and \( d_i \) is the ground truth depth.




# Awesome-Depth-Estimation
A curated list of papers and resources focused on Depth Estimation. 

## Table of Contents

1. [Selected Papers on Depth Estimation](#selected-papers-on-depth-estimation)
2. [Useful Datasets](#useful-datasets)
3. [Evaluation Metrics](#evaluation-metrics)

---

## 1. Selected Papers on Depth Estimation

### **UniDepth: Universal Monocular Metric Depth Estimation**  

Authors: Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu  

Published: CVPR 2024

**[Paper](https://arxiv.org/pdf/2403.18913)** | **[Project](https://lpiccinelli-eth.github.io/pub/unidepth/)** | **[Code](https://github.com/lpiccinelli-eth/UniDepth)**

***Keywords***: *Monocular Depth Estimation*, *Transformer based*, *Images Depth*

<details>
  <summary>Click to view Abstract</summary>

  Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepth, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE methods, UniDepth directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepth implements a self-promptable camera module predicting dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. Thorough evaluations on ten datasets in a zero-shot regime consistently demonstrate the superior performance of UniDepth, even when compared with methods directly trained on the testing domains.

</details>

### **DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos**  

Authors: Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, Ying Shan  

Published: CVPR 2024

**[Paper](https://arxiv.org/pdf/2409.02095)** | **[Project](https://depthcrafter.github.io/)** | **[Code](https://github.com/Tencent/DepthCrafter)**

***Keywords***: *Video Depth Estimation*, *Open-world Videos*, *Consistency*, *Depth Sequences*

<details>
  <summary>Click to view Abstract</summary>

  Estimating video depth in open-world scenarios is challenging due to the diversity of videos in appearance, content motion, camera movement, and length. We present DepthCrafter, an innovative method for generating temporally consistent long depth sequences with intricate details for open-world videos, without requiring any supplementary information such as camera poses or optical flow. The generalization ability to open-world videos is achieved by training the video-to-depth model from a pretrained image-to-video diffusion model, through our meticulously designed three-stage training strategy. Our training approach enables the model to generate depth sequences with variable lengths at one time, up to 110 frames, and harvest both precise depth details and rich content diversity from realistic and synthetic datasets. We also propose an inference strategy that can process extremely long videos through segment-wise estimation and seamless stitching. Comprehensive evaluations on multiple datasets reveal that DepthCrafter achieves state-of-the-art performance in open-world video depth estimation under zero-shot settings. Furthermore, DepthCrafter facilitates various downstream applications, including depth-based visual effects and conditional video generation.
  
</details>



## 2. Useful Datasets

Here is a list of useful datasets for depth estimation:

### **NYU Depth Dataset V2**  
Authors: Nathan Silberman, Pushmeet Kohli, Derek Hoiem, Rob Fergus  
Published: ECCV 2012

**[Paper](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf)** | **[Project](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)**

### **ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes**  
Authors: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner  

**[Paper](http://www.scan-net.org/)** | **[Project](https://github.com/ScanNet/ScanNet)** | **[Code](https://github.com/ScanNet/ScanNet)**

### **SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels**  
Authors: Jianxiong Xiao, Andrew Owens, Antonio Torralba  
Published: Proceedings of 14th IEEE International Conference on Computer Vision (ICCV2013)

**[Paper](https://vision.princeton.edu/projects/2013/SUN3D/paper.pdf)** | **[Project](https://sun3d.cs.princeton.edu/)**

### **KITTI**  
Authors: Andreas Geiger, Philip Lenz, Christoph Stiller, Raquel Urtasun  

**[Project](https://www.cvlibs.net/datasets/kitti/)**

### **DDAD-Dense Depth for Autonomous Driving**  
Authors: Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, Adrien Gaidon  
Published: CVPR 2020  

**[Project](https://github.com/TRI-ML/DDAD)**

### **DIODE: A Dense Indoor and Outdoor DEpth Dataset**  
Authors: Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, Gregory Shakhnarovich  

**[Project](https://diode-dataset.org/)** | **[Paper](https://arxiv.org/pdf/1908.00463)** | **[Code](https://github.com/diode-dataset/diode-devkit)**

### **Hypersim**  
Authors: Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, Joshua M. Susskind  
Published: ICCV 2021

**[Project](https://github.com/apple/ml-hypersim)**



---

## 3. Evaluation Metrics

To evaluate depth estimation models, the following metrics are commonly used:

1. **AbsRel (Absolute Relative Error)**:
   - **Definition**: Measures the average relative error between predicted depth and ground truth depth.
   - **Formula**:
     ```
     AbsRel = (1 / N) * Σ |d_hat_i - d_i| / d_i
     ```
     Where `d_hat_i` is the predicted depth, `d_i` is the ground truth depth, and `N` is the number of pixels or points.

2. **δ₁ (Delta 1)**:
   - **Definition**: Measures the percentage of predictions where the predicted depth is within a factor of 1.0 of the ground truth depth.
   - **Formula**:
     ```
     δ₁ = (1 / N) * Σ 1{d_hat_i / d_i < 1.0}
     ```
     Where `d_hat_i` is the predicted depth, `d_i` is the ground truth depth, and the indicator function equals 1 if the condition is true, otherwise 0.

3. **RMS (Root Mean Squared Error)**:
   - **Definition**: Measures the square root of the average squared differences between predicted and ground truth depths.
   - **Formula**:
     ```
     RMS = sqrt((1 / N) * Σ (d_hat_i - d_i)²)
     ```
     Where `d_hat_i` is the predicted depth, `d_i` is the ground truth depth, and `N` is the number of pixels or points.

4. **RMSlog (Logarithmic Root Mean Squared Error)**:
   - **Definition**: Similar to RMS, but applied to the logarithms of the depth values to focus on the relative differences.
   - **Formula**:
     ```
     RMS_log = sqrt((1 / N) * Σ (log(d_hat_i) - log(d_i))²)
     ```
     Where `log` is the logarithmic transformation.

5. **CD (Chernoff Distance)**:
   - **Definition**: Measures the difference between predicted and ground truth depth distributions.
   - **Formula**: Typically computed using statistical distance measures such as KL divergence or other distribution-based methods.

6. **SIlog (Scaled Logarithmic Error)**:
   - **Definition**: Evaluates the error in the logarithmic scale of depth values, scaled to the range of depths.
   - **Formula**:
     ```
     SI_log = (1 / N) * Σ |log(d_hat_i) - log(d_i)|
     ```
     Where `d_hat_i` is the predicted depth and `d_i` is the ground truth depth.


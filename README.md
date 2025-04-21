# Awesome-Depth-Estimation
A curated list of papers and resources focused on Depth Estimation. 

## Table of Contents  

1. [Survey](#1-survey)  
2. [Monocular Depth Estimation](#2-monocular-depth-estimation)  
3. [Others (Fisheye, 360-degree)](#3-others-fisheye-360-degree)

[Useful Datasets](#useful-datasets)  
[Evaluation Metrics](#evaluation-metrics)  

---

## 1. Survey

+ 📄 **A Systematic Literature Review on Deep Learning-based Depth Estimation in Computer Vision**  
   arXiv 2025 | [Paper](https://arxiv.org/pdf/2501.05147) | *Keywords: Deep Learning, Depth Estimation, Monocular/Stereo/Multi-view*

+ 📄 **Deep Learning-based Depth Estimation Methods from Monocular Image and Videos: A Comprehensive Survey**  
   arXiv 2024 | [Paper](https://arxiv.org/pdf/2406.19675) | *Keywords: Monocular Depth, Video Depth, Deep Learning Survey*

+ 📄 **A Study on the Generality of Neural Network Structures for Monocular Depth Estimation**  
   TPAMI 2023 | [Paper](https://arxiv.org/pdf/2301.03169) | *Keywords: CNN/Transformer, Generalization, Shape Bias*

+ 📄 **Deep Digging into the Generalization of Self-Supervised Monocular Depth Estimation**  
   2022 | [Paper](https://arxiv.org/pdf/2205.11083) | *Keywords: Self-Supervised, Hybrid Models, Generalization*

+ 📄 **Monocular Depth Estimation Using Deep Learning: A Review**  
   2022 | [Paper](https://www.mdpi.com/1424-8220/22/14/5353) | *Keywords: Robotics, Autonomous Vehicles, AR/VR*

+ 📄 **Outdoor Monocular Depth Estimation: A Research Review**  
   arXiv 2022 | [Paper](https://arxiv.org/pdf/2205.01399) | *Keywords: Outdoor Scenes, Domain Adaptation*

+ 📄 **Deep Learning for Monocular Depth Estimation: A Review**  
   2021 | [Paper](https://pure.port.ac.uk/ws/files/26286067/Deep_Learning_for_Monocular_Depth_Estimation_A_Review_pp.pdf) | *Keywords: Augmented Reality, 3D Reconstruction*

+ 📄 **A Survey of Depth Estimation Based on Computer Vision**  
   DSC 2020 | [Paper](https://ieeexplore.ieee.org/document/9172861) | *Keywords: 3D Perception, SLAM*

+ 📄 **Monocular Depth Estimation Based On Deep Learning: An Overview**  
   arXiv 2020 | [Paper](https://arxiv.org/pdf/2003.06620) | *Keywords: End-to-End Learning, Multi-task*

+ 📄 **Monocular Depth Estimation: A Survey**  
   2019 | [Paper](https://arxiv.org/pdf/1901.09402) | *Keywords: Ill-posed Problem, Scene Understanding*


---

## 2. Monocular Depth Estimation

### 2.1 Metric Depth



+ **📄 UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler**

   Authors: Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, Luc Van Gool

   Published: arXiv 2024

   [![Paper](https://img.shields.io/badge/arXiv-2502.20110-b31b1b.svg)](https://arxiv.org/abs/2502.20110)
   [![Code](https://img.shields.io/github/stars/lpiccinelli-eth/UniDepth.svg?style=social&label=Star)](https://github.com/lpiccinelli-eth/UniDepth)

   <details>
   <summary>Click to view Abstract</summary>

   Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepthV2, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE paradigm, UniDepthV2 directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepthV2 implements a self-promptable camera module predicting a dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles the camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. UniDepthV2 improves its predecessor UniDepth model via a new edge-guided loss which enhances the localization and sharpness of edges in the metric depth outputs, a revisited, simplified and more efficient architectural design, and an additional uncertainty-level output which enables downstream tasks requiring confidence. Thorough evaluations on ten depth datasets in a zero-shot regime consistently demonstrate the superior performance and generalization of UniDepthV2.

   </details>

+ **📄 UniDepth: Universal Monocular Metric Depth Estimation**

   Authors: Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2403.18913-b31b1b.svg)](https://arxiv.org/pdf/2403.18913)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://lpiccinelli-eth.github.io/pub/unidepth/)
   [![Code](https://img.shields.io/github/stars/lpiccinelli-eth/UniDepth.svg?style=social&label=Star)](https://github.com/lpiccinelli-eth/UniDepth)

   <details>
   <summary>Click to view Abstract</summary>

   Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepth, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE methods, UniDepth directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepth implements a self-promptable camera module predicting dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. Thorough evaluations on ten datasets in a zero-shot regime consistently demonstrate the superior performance of UniDepth, even when compared with methods directly trained on the testing domains.

   </details>



+ **📄 DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos**

   Authors: Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, Ying Shan

   Published: arXiv 2024

   [![Paper](https://img.shields.io/badge/arXiv-2409.02095-b31b1b.svg)](https://arxiv.org/pdf/2409.02095)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://depthcrafter.github.io/)
   [![Code](https://img.shields.io/github/stars/Tencent/DepthCrafter.svg?style=social&label=Star)](https://github.com/Tencent/DepthCrafter)

   <details>
   <summary>Click to view Abstract</summary>

   Estimating video depth in open-world scenarios is challenging due to the diversity of videos in appearance, content motion, camera movement, and length. We present DepthCrafter, an innovative method for generating temporally consistent long depth sequences with intricate details for open-world videos, without requiring any supplementary information such as camera poses or optical flow. The generalization ability to open-world videos is achieved by training the video-to-depth model from a pretrained image-to-video diffusion model, through our meticulously designed three-stage training strategy. Our training approach enables the model to generate depth sequences with variable lengths at one time, up to 110 frames, and harvest both precise depth details and rich content diversity from realistic and synthetic datasets. We also propose an inference strategy that can process extremely long videos through segment-wise estimation and seamless stitching. Comprehensive evaluations on multiple datasets reveal that DepthCrafter achieves state-of-the-art performance in open-world video depth estimation under zero-shot settings. Furthermore, DepthCrafter facilitates various downstream applications, including depth-based visual effects and conditional video generation.

   </details>



+ **📄 Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation**  

   Authors: Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, Shaojie Shen  

   Published: TPAMI 2024

   [![Paper](https://img.shields.io/badge/arXiv-2404.15506-b31b1b.svg)](https://arxiv.org/pdf/2404.15506)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://jugghm.github.io/Metric3Dv2/)
   [![Code](https://img.shields.io/github/stars/YvanYin/Metric3D.svg?style=social&label=Star)](https://github.com/YvanYin/Metric3D?tab=readme-ov-file)

   <details>
   <summary>Click to view Abstract</summary>

   We introduce Metric3D v2, a geometric foundation model for zero-shot metric depth and surface normal estimation from a single image, which is crucial for metric 3D recovery. While depth and normal are geometrically related and highly complimentary, they present distinct challenges. State-of-the-art (SoTA) monocular depth methods achieve zero-shot generalization by learning affine-invariant depths, which cannot recover real-world metrics. Meanwhile, SoTA normal estimation methods have limited zero-shot performance due to the lack of large-scale labeled data. To tackle these issues, we propose solutions for both metric depth estimation and surface normal estimation. For metric depth estimation, we show that the key to a zero-shot single-view model lies in resolving the metric ambiguity from various camera models and large-scale data training. We propose a canonical camera space transformation module, which explicitly addresses the ambiguity problem and can be effortlessly plugged into existing monocular models. For surface normal estimation, we propose a joint depth-normal optimization module to distill diverse data knowledge from metric depth, enabling normal estimators to learn beyond normal labels. Equipped with these modules, our depth-normal models can be stably trained with over 16 million of images from thousands of camera models with different-type annotations, resulting in zero-shot generalization to in-the-wild images with unseen camera settings. Our method currently ranks the 1st on various zero-shot and non-zero-shot benchmarks for metric depth, affine-invariant-depth as well as surface-normal prediction. Notably, we surpassed the ultra-recent MarigoldDepth and DepthAnything on various depth benchmarks including NYUv2 and KITTI. Our method enables the accurate recovery of metric 3D structures on randomly collected internet images, paving the way for plausible single-image metrology. The potential benefits extend to downstream tasks, which can be significantly improved by simply plugging in our model. For example, our model relieves the scale drift issues of monocular-SLAM, leading to high-quality metric scale dense mapping. These applications highlight the versatility of Metric3D v2 models as geometric foundation models.
  
   </details>




+ **📄 Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image**  

   Authors: Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen  

   Published: ICCV 2023

   [![Paper](https://img.shields.io/badge/arXiv-2307.10984-b31b1b.svg)](https://arxiv.org/pdf/2307.10984)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://jugghm.github.io/Metric3Dv2/)
   [![Code](https://img.shields.io/github/stars/YvanYin/Metric3D.svg?style=social&label=Star)](https://github.com/YvanYin/Metric3D?tab=readme-ov-file)

   <details>
   <summary>Click to view Abstract</summary>

   Reconstructing accurate 3D scenes from images is a long-standing vision task. Due to the ill-posedness of the single-image reconstruction problem, most well-established methods are built upon multi-view geometry. State-of-the-art (SOTA) monocular metric depth estimation methods can only handle a single camera model and are unable to perform mixed-data training due to the metric ambiguity. Meanwhile, SOTA monocular methods trained on large mixed datasets achieve zero-shot generalization by learning affine-invariant depths, which cannot recover real-world metrics. In this work, we show that the key to a zero-shot single-view metric depth model lies in the combination of large-scale data training and resolving the metric ambiguity from various camera models. We propose a canonical camera space transformation module, which explicitly addresses the ambiguity problems and can be effortlessly plugged into existing monocular models. Equipped with our module, monocular models can be stably trained over 8 million of images with thousands of camera models, resulting in zero-shot generalization to in-the-wild images with unseen camera settings. Experiments demonstrate SOTA performance of our method on 7 zero-shot benchmarks. Notably, our method won the championship in the 2nd Monocular Depth Estimation Challenge. Our method enables the accurate recovery of metric 3D structures on randomly collected internet images, paving the way for plausible single-image metrology. The potential benefits extend to downstream tasks, which can be significantly improved by simply plugging in our model. For example, our model relieves the scale drift issues of monocular-SLAM, leading to high-quality metric scale dense mapping.
   </details>




### 2.2 Relative Depth

+ **📄 VGGT: Visual Geometry Grounded Transformer**

   Authors: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny
  
   Published: CVPR 2025 oral  

   [![Paper](https://img.shields.io/badge/arXiv-2503.11651-b31b1b.svg)](https://arxiv.org/abs/2503.11651)
   [![Code](https://img.shields.io/github/stars/facebookresearch/vggt.svg?style=social&label=Star)](https://github.com/facebookresearch/vggt)

   <details>
   <summary>Click to view Abstract</summary>

   We present VGGT, a feed-forward neural network that directly infers all key 3D attributes of a scene, including camera parameters, point maps, depth maps, and 3D point tracks, from one, a few, or hundreds of its views. This approach is a step forward in 3D computer vision, where models have typically been constrained to and specialized for single tasks. It is also simple and efficient, reconstructing images in under one second, and still outperforming alternatives that require post-processing with visual geometry optimization techniques. The network achieves state-of-the-art results in multiple 3D tasks, including camera parameter estimation, multi-view depth estimation, dense point cloud reconstruction, and 3D point tracking. We also show that using pretrained VGGT as a feature backbone significantly enhances downstream tasks, such as non-rigid point tracking and feed-forward novel view synthesis.

   </details>



+ **📄 DUSt3R: Geometric 3D Vision Made Easy**

   Authors: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.14132)
   [![Project](https://img.shields.io/badge/Project-Page-00CED1)](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)
   [![Code](https://img.shields.io/github/stars/naver/dust3r.svg?style=social&label=Star)](https://github.com/naver/dust3r)

   <details>
   <summary>Click to view Abstract</summary>

   Multi-view stereo reconstruction (MVS) in the wild requires estimating camera parameters, such as intrinsic and extrinsic parameters, which are typically tedious and cumbersome to obtain. These parameters are essential for triangulating corresponding pixels in 3D space, a core aspect of high-performing MVS algorithms. In this work, we introduce DUSt3R, a novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, operating without prior information about camera calibration or viewpoint poses. We reformulate the pairwise reconstruction problem as a regression of pointmaps, relaxing the constraints of traditional projective camera models. This formulation unifies monocular and binocular reconstruction cases. For scenarios with multiple images, we propose a simple yet effective global alignment strategy that aligns all pairwise pointmaps in a common reference frame. Our network architecture is based on standard Transformer encoders and decoders, leveraging powerful pretrained models. DUSt3R provides a direct 3D model of the scene, depth information, and seamlessly recovers pixel matches, relative, and absolute cameras. Extensive experiments demonstrate that DUSt3R unifies various 3D vision tasks and sets new SoTAs in monocular/multi-view depth estimation and relative pose estimation. In summary, DUSt3R simplifies many geometric 3D vision tasks.

   </details>


### 2.3 Depth Completion


+ **📄 DepthLab: From Partial to Complete**  

   Authors: Zhiheng Liu, Ka Leong Cheng, Qiuyu Wang, Shuzhe Wang, Hao Ouyang, Bin Tan, Kai Zhu, Yujun Shen, Qifeng Chen, Ping Luo  

   Published: arXiv 2024

   [![Paper](https://img.shields.io/badge/arXiv-2412.18153-b31b1b.svg)](https://arxiv.org/pdf/2412.18153)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://johanan528.github.io/depthlab_web/)
   [![Code](https://img.shields.io/github/stars/ant-research/DepthLab.svg?style=social&label=Star)](https://github.com/ant-research/DepthLab)

   <details>
   <summary>Click to view Abstract</summary>

   Missing values remain a common challenge for depth data across its wide range of applications, stemming from various causes like incomplete data acquisition and perspective alteration. This work bridges this gap with DepthLab, a foundation depth inpainting model powered by image diffusion priors. Our model features two notable strengths: (1) it demonstrates resilience to depth-deficient regions, providing reliable completion for both continuous areas and isolated points, and (2) it faithfully preserves scale consistency with the conditioned known depth when filling in missing values. Drawing on these advantages, our approach proves its worth in various downstream tasks, including 3D scene inpainting, text-to-3D scene generation, sparse-view reconstruction with DUST3R, and LiDAR depth completion, exceeding current solutions in both numerical performance and visual quality.

   </details>



---

## 3 Others (Fisheye, 360-degree)


---

## Useful Datasets

Here is a list of useful datasets for depth estimation:

### **📦 NYU Depth Dataset V2**  
Authors: Nathan Silberman, Pushmeet Kohli, Derek Hoiem, Rob Fergus  
Published: ECCV 2012

**[Paper](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf)** | **[Project](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)**

### **📦 ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes**  
Authors: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Nießner  

**[Paper](http://www.scan-net.org/)** | **[Project](https://github.com/ScanNet/ScanNet)** | **[Code](https://github.com/ScanNet/ScanNet)**

### **📦 SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels**  
Authors: Jianxiong Xiao, Andrew Owens, Antonio Torralba  
Published: Proceedings of 14th IEEE International Conference on Computer Vision (ICCV2013)

**[Paper](https://vision.princeton.edu/projects/2013/SUN3D/paper.pdf)** | **[Project](https://sun3d.cs.princeton.edu/)**

### **📦 KITTI**  
Authors: Andreas Geiger, Philip Lenz, Christoph Stiller, Raquel Urtasun  

**[Project](https://www.cvlibs.net/datasets/kitti/)**

### **📦 DDAD-Dense Depth for Autonomous Driving**  
Authors: Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, Adrien Gaidon  
Published: CVPR 2020  

**[Project](https://github.com/TRI-ML/DDAD)**

### **📦 DIODE: A Dense Indoor and Outdoor DEpth Dataset**  
Authors: Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, Gregory Shakhnarovich  

**[Project](https://diode-dataset.org/)** | **[Paper](https://arxiv.org/pdf/1908.00463)** | **[Code](https://github.com/diode-dataset/diode-devkit)**

### **📦 Hypersim**  
Authors: Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, Joshua M. Susskind  
Published: ICCV 2021

**[Project](https://github.com/apple/ml-hypersim)**



---

## Evaluation Metrics

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


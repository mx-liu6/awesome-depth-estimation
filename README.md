# Awesome-Depth-Estimation
A curated list of papers and resources focused on Depth Estimation. 

## Table of Contents  

1. [Survey](#1-survey)  
2. [Monocular Depth Estimation](#2-monocular-depth-estimation)
    - [Metric Depth](#21-metric-depth)
    - [Relative Depth](#22-relative-depth)      
    - [Depth Completion](#23-depth-completion)  
3. [Others (Fisheye, 360-degree)](#3-others-fisheye-360-degree)

[Useful Datasets](#useful-datasets)  
[Evaluation Metrics](#evaluation-metrics)  

---

## 1. Survey

#### **📄 A Systematic Literature Review on Deep Learning-based Depth Estimation in Computer Vision**

Authors: Ali Rohana, Md Junayed Hasana, Andrei Petrovskia

Published: arXiv 2025

**[Paper](https://arxiv.org/pdf/2501.05147)**

***Keywords***: *Deep Learning (DL)*, *Artificial Intelligence (AI)*, *Depth Estimation*, *Monocular Depth Estimation*, *Stereo Depth Estimation*, *Multi-view*

<details>
  <summary>Click to view Abstract</summary>

  Depth estimation (DE) provides spatial information about a scene and enables tasks such as 3D reconstruction, object detection, and scene understanding. Recently, there has been an increasing interest in using deep learning (DL)-based methods for DE. Traditional techniques rely on handcrafted features that often struggle to generalize to diverse scenes and require extensive manual tuning. However, DL models for DE can automatically extract relevant features from input data, adapt to various scene conditions, and generalize well to unseen environments. Numerous DL-based methods have been developed, making it necessary to survey and synthesize the state-of-the-art (SOTA). Previous reviews on DE have mainly focused on either monocular or stereo-based techniques, rather than comprehensively reviewing DE. Furthermore, to the best of our knowledge, there is no systematic literature review (SLR) that comprehensively focuses on DE. Therefore, this SLR study is being conducted. Initially, electronic databases were searched for relevant publications, resulting in 1284 publications. Using defined exclusion and quality criteria, 128 publications were shortlisted and further filtered to select 59 high-quality primary studies. These studies were analyzed to extract data and answer defined research questions. Based on the results, DL methods were developed for mainly three different types of DE: monocular, stereo, and multi-view. 20 publicly available datasets were used to train, test, and evaluate DL models for DE, with KITTI, NYU Depth V2, and Make 3D being the most used datasets. 29 evaluation metrics were used to assess the performance of DE. 35 base models were reported in the primary studies, and the top five most-used base models were ResNet-50, ResNet-18, ResNet-101, U-Net, and VGG-16. Finally, the lack of ground truth data was among the most significant challenges reported by primary studies.
  
</details>


#### **📄 Deep Learning-based Depth Estimation Methods from Monocular Image and Videos: A Comprehensive Survey**  

Authors: Uchitha Rajapaksha, Ferdous Sohel, Hamid Laga, Dean Diepeveen, Mohammed Bennamoun

Published: arXiv 2024 

**[Paper](https://arxiv.org/pdf/2406.19675)**  

***Keywords***: *Depth Estimation*, *Monocular Images*, *Monocular Videos*, *Deep Learning*, *Survey*

<details>
  <summary>Click to view Abstract</summary>

  Estimating depth from single RGB images and videos is of widespread interest due to its applications in many areas, including autonomous driving, 3D reconstruction, digital entertainment, and robotics. More than 500 deep learning-based papers have been published in the past 10 years, which indicates the growing interest in the task. This paper presents a comprehensive survey of the existing deep learning-based methods, the challenges they address, and how they have evolved in their architecture and supervision methods. It provides a taxonomy for classifying the current work based on their input and output modalities, network architectures, and learning methods. It also discusses the major milestones in the history of monocular depth estimation, and different pipelines, datasets, and evaluation metrics used in existing methods.
  
</details>


#### **📄 A Study on the Generality of Neural Network Structures for Monocular Depth Estimation**

Authors: Jinwoo Bae, Kyumin Hwang, Sunghoon Im

Published: TPAMI 2023

**[Paper](https://arxiv.org/pdf/2301.03169)**

***Keywords***: *Monocular Depth Estimation*, *Generalization*, *CNN*, *Transformer*, *Out-of-Distribution*, *Ablation Study*

<details>
  <summary>Click to view Abstract</summary>

  Monocular depth estimation has been widely studied, and significant improvements in performance have been recently reported. However, most previous works are evaluated on a few benchmark datasets, such as KITTI datasets, and none of the works provide an in-depth analysis of the generalization performance of monocular depth estimation. In this paper, we deeply investigate the various backbone networks (e.g. CNN and Transformer models) toward the generalization of monocular depth estimation. First, we evaluate state-of-the-art models on both in-distribution and out-of-distribution datasets, which have never been seen during network training. Then, we investigate the internal properties of the representations from the intermediate layers of CNN-/Transformer-based models using synthetic texture-shifted datasets. Through extensive experiments, we observe that the Transformers exhibit a strong shape-bias rather than CNNs, which have a strong texture-bias. We also discover that texture-biased models exhibit worse generalization performance for monocular depth estimation than shape-biased models. We demonstrate that similar aspects are observed in real-world driving datasets captured under diverse environments. Lastly, we conduct a dense ablation study with various backbone networks which are utilized in modern strategies. The experiments demonstrate that the intrinsic locality of the CNNs and the self-attention of the Transformers induce texture-bias and shape-bias, respectively.
  
</details>


#### **📄 Deep Digging into the Generalization of Self-Supervised Monocular Depth Estimation**  

Authors: Jinwoo Bae, Sungho Moon, Sunghoon Im

Published: 2022 

**[Paper](https://arxiv.org/pdf/2205.11083)**  

***Keywords***: *Self-Supervised Learning*, *Monocular Depth Estimation*, *Generalization*, *CNN*, *Transformers*, *Hybrid Models*, *MonoFormer*

<details>
  <summary>Click to view Abstract</summary>

  Self-supervised monocular depth estimation has been widely studied recently. Most of the work has focused on improving performance on benchmark datasets, such as KITTI, but has offered a few experiments on generalization performance. In this paper, we investigate the backbone networks (e.g. CNNs, Transformers, and CNN-Transformer hybrid models) toward the generalization of monocular depth estimation. We first evaluate state-of-the-art models on diverse public datasets, which have never been seen during the network training. Next, we investigate the effects of texture-biased and shape-biased representations using the various texture-shifted datasets that we generated. We observe that Transformers exhibit a strong shape bias and CNNs do a strong texture-bias. We also find that shape-biased models show better generalization performance for monocular depth estimation compared to texture-biased models. Based on these observations, we newly design a CNN-Transformer hybrid network with a multi-level adaptive feature fusion module, called MonoFormer. The design intuition behind MonoFormer is to increase shape bias by employing Transformers while compensating for the weak locality bias of Transformers by adaptively fusing multi-level representations. Extensive experiments show that the proposed method achieves state-of-the-art performance with various public datasets. Our method also shows the best generalization ability among the competitive methods.
  
</details>


#### **📄 Monocular Depth Estimation Using Deep Learning: A Review**  

Authors: Armin Masoumian, Hatem A. Rashwan, Julián Cristiano, M. Salman Asif, Domenec Puig

Published: 2022 

**[Paper](https://www.mdpi.com/1424-8220/22/14/5353)**  

***Keywords***: *Monocular Depth Estimation*, *Deep Learning*, *Robotics*, *Autonomous Vehicles*, *Augmented Reality*

<details>
  <summary>Click to view Abstract</summary>

  In current decades, significant advancements in robotics engineering and autonomous vehicles have improved the requirement for precise depth measurements. Depth estimation (DE) is a traditional task in computer vision that can be appropriately predicted by applying numerous procedures. This task is vital in disparate applications such as augmented reality and target tracking. Conventional monocular DE (MDE) procedures are based on depth cues for depth prediction. Various deep learning techniques have demonstrated their potential applications in managing and supporting the traditional ill-posed problem. The principal purpose of this paper is to represent a state-of-the-art review of the current developments in MDE based on deep learning techniques. For this goal, this paper tries to highlight the critical points of the state-of-the-art works on MDE from disparate aspects. These aspects include input data shapes and training manners such as supervised, semi-supervised, and unsupervised learning approaches in combination with applying different datasets and evaluation indicators. At last, limitations regarding the accuracy of the DL-based MDE models, computational time requirements, real-time inference, transferability, input images shape and domain adaptation, and generalization are discussed to open new directions for future research.
  
</details>

#### **📄 Outdoor Monocular Depth Estimation: A Research Review**

Authors: Pulkit Vyas, Chirag Saxena, Anwesh Badapanda, Anurag Goswami

Published: arXiv 2022

**[Paper](https://arxiv.org/pdf/2205.01399)**

***Keywords***: *Monocular Depth Estimation*, *Outdoor Dataset*, *Deep Learning*

<details>
  <summary>Click to view Abstract</summary>

  Depth estimation is an important task, applied in various methods and applications of computer vision. While the traditional methods of estimating depth are based on depth cues and require specific equipment such as stereo cameras and configuring input according to the approach being used, the focus at the current time is on a single source, or monocular, depth estimation. The recent developments in Convolution Neural Networks along with the integration of classical methods in these deep learning approaches have led to a lot of advancements in the depth estimation problem. The problem of outdoor depth estimation, or depth estimation in wild, is a very scarcely researched field of study. In this paper, we give an overview of the available datasets, depth estimation methods, research work, trends, challenges, and opportunities that exist for open research. To our knowledge, no openly available survey work provides a comprehensive collection of outdoor depth estimation techniques and research scope, making our work an essential contribution for people looking to enter this field of study.
  
</details>


#### **📄 Deep Learning for Monocular Depth Estimation: A Review**  

Authors: Yue Ming, Xuyang Meng, Chunxiao Fan, Hui Yu

Published: 2021 

**[Paper](https://pure.port.ac.uk/ws/files/26286067/Deep_Learning_for_Monocular_Depth_Estimation_A_Review_pp.pdf)**  

***Keywords***: *Monocular Depth Estimation*, *Deep Learning*, *Augmented Reality*, *Autonomous Driving*, *Target Tracking*

<details>
  <summary>Click to view Abstract</summary>

  Depth estimation is a classic task in computer vision, which is of great significance for many applications such as augmented reality, target tracking, and autonomous driving. Traditional monocular depth estimation methods are based on depth cues for depth prediction with strict requirements, e.g., shape-from-focus/defocus methods require low depth of field on the scenes and images. Recently, a large body of deep learning methods has been proposed and shown great promise in handling the traditional ill-posed problem. This paper aims to review the state-of-the-art development in deep learning-based monocular depth estimation. We give an overview of published papers between 2014 and 2020 in terms of training manners and task types. We firstly summarize the deep learning models for monocular depth estimation. Secondly, we categorize various deep learning-based methods in monocular depth estimation. Thirdly, we introduce the publicly available dataset and the evaluation metrics. And we also analyze the properties of these methods and compare their performance. Finally, we highlight the challenges in order to inform the future research directions.
  
</details>


#### **📄 A Survey of Depth Estimation Based on Computer Vision**

Authors: Yang Liu, Jie Jiang, Jiahao Sun, Liang Bai, Qi Wang

Published: DSC 2020

**[Paper](https://ieeexplore.ieee.org/document/9172861)**

***Keywords***: *Computer Vision*, *Depth Estimation*, *Pose Estimation*, *3D Reconstruction*

<details>
  <summary>Click to view Abstract</summary>

  Currently, the method based on computer vision for depth information extraction and depth estimation is widely used. It can get depth information from 2D images, depth maps, or binocular vision images and has been a popular application in the field of artificial intelligence such as depth detection, pose estimation, as well as 3D reconstruction. This paper introduces the basic theory and some implementation methods of depth information acquisition based on computer vision. As well, it briefly summarizes the existing research results and makes an outlook on the future development trend of the field.
  
</details>


#### **📄 Monocular Depth Estimation Based On Deep Learning: An Overview**  

Authors: Chaoqiang Zhao, Qiyu Sun, Chongzhen Zhang, Yang Tang, Feng Qian

Published: arXiv 2020 

**[Paper](https://arxiv.org/pdf/2003.06620)**  

***Keywords***: *Monocular Depth Estimation*, *Deep Learning*, *Single Image Depth*, *End-to-End Networks*, *Supervised/Unsupervised/Semi-supervised Learning*

<details>
  <summary>Click to view Abstract</summary>

  Depth information is important for autonomous systems to perceive environments and estimate their own state. Traditional depth estimation methods, like structure from motion and stereo vision matching, are built on feature correspondences of multiple viewpoints. Meanwhile, the predicted depth maps are sparse. Inferring depth information from a single image (monocular depth estimation) is an ill-posed problem. With the rapid development of deep neural networks, monocular depth estimation based on deep learning has been widely studied recently and achieved promising performance in accuracy. Meanwhile, dense depth maps are estimated from single images by deep neural networks in an end-to-end manner. In order to improve the accuracy of depth estimation, different kinds of network frameworks, loss functions, and training strategies are proposed subsequently. Therefore, we survey the current monocular depth estimation methods based on deep learning in this review. Initially, we conclude several widely used datasets and evaluation indicators in deep learning-based depth estimation. Furthermore, we review some representative existing methods according to different training manners: supervised, unsupervised, and semi-supervised. Finally, we discuss the challenges and provide some ideas for future research in monocular depth estimation.
  
</details>


#### **📄 Monocular Depth Estimation: A Survey**  

Authors: Amlaan Bhoi

Published: 2019 

**[Paper](https://arxiv.org/pdf/1901.09402)**  

***Keywords***: *Monocular Depth Estimation*, *Supervised Learning*, *Weakly-supervised Learning*, *Unsupervised Learning*, *Scene Reconstruction*

<details>
  <summary>Click to view Abstract</summary>

  Monocular depth estimation is often described as an ill-posed and inherently ambiguous problem. Estimating depth from 2D images is a crucial step in scene reconstruction, 3D object recognition, segmentation, and detection. The problem can be framed as: given a single RGB image as input, predict a dense depth map for each pixel. This problem is worsened by the fact that most scenes have large texture and structural variations, object occlusions, and rich geometric detailing. All these factors contribute to difficulty in accurate depth estimation. In this paper, we review five papers that attempt to solve the depth estimation problem with various techniques including supervised, weakly-supervised, and unsupervised learning techniques. We then compare these papers and understand the improvements made over one another. Finally, we explore potential improvements that can aid to better solve this problem.
  
</details>


---

## 2. Monocular Depth Estimation

### 2.1 Metric Depth

#### **📄 UniDepth: Universal Monocular Metric Depth Estimation**  

Authors: Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu  

Published: CVPR 2024

**[Paper](https://arxiv.org/pdf/2403.18913)** | **[Project](https://lpiccinelli-eth.github.io/pub/unidepth/)** | **[Code](https://github.com/lpiccinelli-eth/UniDepth)**

***Keywords***: *Monocular Depth Estimation*, *Transformer based*, *Images Depth*

<details>
  <summary>Click to view Abstract</summary>

  Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepth, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE methods, UniDepth directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepth implements a self-promptable camera module predicting dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. Thorough evaluations on ten datasets in a zero-shot regime consistently demonstrate the superior performance of UniDepth, even when compared with methods directly trained on the testing domains.

</details>




#### **📄 DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos**  

Authors: Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, Ying Shan  

Published: arXiv 2024

**[Paper](https://arxiv.org/pdf/2409.02095)** | **[Project](https://depthcrafter.github.io/)** | **[Code](https://github.com/Tencent/DepthCrafter)**

***Keywords***: *Video Depth Estimation*, *Open-world Videos*, *Consistency*, *Depth Sequences*

<details>
  <summary>Click to view Abstract</summary>

  Estimating video depth in open-world scenarios is challenging due to the diversity of videos in appearance, content motion, camera movement, and length. We present DepthCrafter, an innovative method for generating temporally consistent long depth sequences with intricate details for open-world videos, without requiring any supplementary information such as camera poses or optical flow. The generalization ability to open-world videos is achieved by training the video-to-depth model from a pretrained image-to-video diffusion model, through our meticulously designed three-stage training strategy. Our training approach enables the model to generate depth sequences with variable lengths at one time, up to 110 frames, and harvest both precise depth details and rich content diversity from realistic and synthetic datasets. We also propose an inference strategy that can process extremely long videos through segment-wise estimation and seamless stitching. Comprehensive evaluations on multiple datasets reveal that DepthCrafter achieves state-of-the-art performance in open-world video depth estimation under zero-shot settings. Furthermore, DepthCrafter facilitates various downstream applications, including depth-based visual effects and conditional video generation.
  
</details>




#### **📄 DepthLab: From Partial to Complete**  

Authors: Zhiheng Liu, Ka Leong Cheng, Qiuyu Wang, Shuzhe Wang, Hao Ouyang, Bin Tan, Kai Zhu, Yujun Shen, Qifeng Chen, Ping Luo  

Published: arXiv 2024

**[Paper](https://arxiv.org/pdf/2412.18153)** | **[Project](https://johanan528.github.io/depthlab_web/)** | **[Code](https://github.com/ant-research/DepthLab)**

***Keywords***: *Depth Inpainting*, *Image Diffusion Priors*, *3D Scene Inpainting*, *LiDAR Depth Completion*

<details>
  <summary>Click to view Abstract</summary>

  Missing values remain a common challenge for depth data across its wide range of applications, stemming from various causes like incomplete data acquisition and perspective alteration. This work bridges this gap with DepthLab, a foundation depth inpainting model powered by image diffusion priors. Our model features two notable strengths: (1) it demonstrates resilience to depth-deficient regions, providing reliable completion for both continuous areas and isolated points, and (2) it faithfully preserves scale consistency with the conditioned known depth when filling in missing values. Drawing on these advantages, our approach proves its worth in various downstream tasks, including 3D scene inpainting, text-to-3D scene generation, sparse-view reconstruction with DUST3R, and LiDAR depth completion, exceeding current solutions in both numerical performance and visual quality.
  
</details>




#### **📄 DUSt3R: Geometric 3D Vision Made Easy**  

Authors: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud  

Published: CVPR 2024

**[Paper](https://arxiv.org/pdf/2312.14132)** | **[Project](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)** | **[Code](https://github.com/naver/dust3r)**

***Keywords***: *Dense 3D Reconstruction*, *Multi-view Stereo*, *Unconstrained Stereo*, *Camera Calibration*

<details>
  <summary>Click to view Abstract</summary>

  Multi-view stereo reconstruction (MVS) in the wild requires to first estimate the camera parameters e.g. intrinsic and extrinsic parameters. These are usually tedious and cumbersome to obtain, yet they are mandatory to triangulate corresponding pixels in 3D space, which is the core of all best performing MVS algorithms. In this work, we take an opposite stance and introduce DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses. We cast the pairwise reconstruction problem as a regression of pointmaps, relaxing the hard constraints of usual projective camera models. We show that this formulation smoothly unifies the monocular and binocular reconstruction cases. In the case where more than two images are provided, we further propose a simple yet effective global alignment strategy that expresses all pairwise pointmaps in a common reference frame. We base our network architecture on standard Transformer encoders and decoders, allowing us to leverage powerful pretrained models. Our formulation directly provides a 3D model of the scene as well as depth information, but interestingly, we can seamlessly recover from it, pixel matches, relative and absolute cameras. Exhaustive experiments on all these tasks showcase that the proposed DUSt3R can unify various 3D vision tasks and set new SoTAs on monocular/multi-view depth estimation as well as relative pose estimation. In summary, DUSt3R makes many geometric 3D vision tasks easy.
  
</details>




#### **📄 Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation**  

Authors: Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, Shaojie Shen  

Published: TPAMI 2024

**[Paper](https://arxiv.org/pdf/2404.15506)** | **[Project](https://jugghm.github.io/Metric3Dv2/)** | **[Code](https://github.com/YvanYin/Metric3D?tab=readme-ov-file)**

***Keywords***: *Zero-shot Metric Depth Estimation*, *Surface Normal Estimation*, *Monocular Geometric Model*, *3D Recovery*

<details>
  <summary>Click to view Abstract</summary>

  We introduce Metric3D v2, a geometric foundation model for zero-shot metric depth and surface normal estimation from a single image, which is crucial for metric 3D recovery. While depth and normal are geometrically related and highly complimentary, they present distinct challenges. State-of-the-art (SoTA) monocular depth methods achieve zero-shot generalization by learning affine-invariant depths, which cannot recover real-world metrics. Meanwhile, SoTA normal estimation methods have limited zero-shot performance due to the lack of large-scale labeled data. To tackle these issues, we propose solutions for both metric depth estimation and surface normal estimation. For metric depth estimation, we show that the key to a zero-shot single-view model lies in resolving the metric ambiguity from various camera models and large-scale data training. We propose a canonical camera space transformation module, which explicitly addresses the ambiguity problem and can be effortlessly plugged into existing monocular models. For surface normal estimation, we propose a joint depth-normal optimization module to distill diverse data knowledge from metric depth, enabling normal estimators to learn beyond normal labels. Equipped with these modules, our depth-normal models can be stably trained with over 16 million of images from thousands of camera models with different-type annotations, resulting in zero-shot generalization to in-the-wild images with unseen camera settings. Our method currently ranks the 1st on various zero-shot and non-zero-shot benchmarks for metric depth, affine-invariant-depth as well as surface-normal prediction. Notably, we surpassed the ultra-recent MarigoldDepth and DepthAnything on various depth benchmarks including NYUv2 and KITTI. Our method enables the accurate recovery of metric 3D structures on randomly collected internet images, paving the way for plausible single-image metrology. The potential benefits extend to downstream tasks, which can be significantly improved by simply plugging in our model. For example, our model relieves the scale drift issues of monocular-SLAM, leading to high-quality metric scale dense mapping. These applications highlight the versatility of Metric3D v2 models as geometric foundation models.
  
</details>




#### **📄 Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image**  

Authors: Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen  

Published: ICCV 2023

**[Paper](https://arxiv.org/pdf/2307.10984)** | **[Project](https://jugghm.github.io/Metric3Dv2/)** | **[Code](https://github.com/YvanYin/Metric3D?tab=readme-ov-file)**

***Keywords***: *Monocular Depth Estimation*, *Zero-shot Generalization*, *Metric 3D Prediction*, *Single Image Reconstruction*

<details>
  <summary>Click to view Abstract</summary>

  Reconstructing accurate 3D scenes from images is a long-standing vision task. Due to the ill-posedness of the single-image reconstruction problem, most well-established methods are built upon multi-view geometry. State-of-the-art (SOTA) monocular metric depth estimation methods can only handle a single camera model and are unable to perform mixed-data training due to the metric ambiguity. Meanwhile, SOTA monocular methods trained on large mixed datasets achieve zero-shot generalization by learning affine-invariant depths, which cannot recover real-world metrics. In this work, we show that the key to a zero-shot single-view metric depth model lies in the combination of large-scale data training and resolving the metric ambiguity from various camera models. We propose a canonical camera space transformation module, which explicitly addresses the ambiguity problems and can be effortlessly plugged into existing monocular models. Equipped with our module, monocular models can be stably trained over 8 million of images with thousands of camera models, resulting in zero-shot generalization to in-the-wild images with unseen camera settings. Experiments demonstrate SOTA performance of our method on 7 zero-shot benchmarks. Notably, our method won the championship in the 2nd Monocular Depth Estimation Challenge. Our method enables the accurate recovery of metric 3D structures on randomly collected internet images, paving the way for plausible single-image metrology. The potential benefits extend to downstream tasks, which can be significantly improved by simply plugging in our model. For example, our model relieves the scale drift issues of monocular-SLAM, leading to high-quality metric scale dense mapping.
</details>




### 2.2 Relative Depth



### 2.3 Depth Completion

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


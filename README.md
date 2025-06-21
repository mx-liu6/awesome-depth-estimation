# Monocular Depth Estimation: A Survey
This repository is relate to our Review Article -- Monocular Depth Estimation: A Survey
We build this repo by the structure from Article, you can just check the link to any paper showned in the article.

## Table of Contents  

1. Introduction and Background 
2. Problem Setup and Datasets 
3. Depth Estimation Prior to Foundation Models 
4. Foundation Models for Monocular Depth Estimation 
5. Downstream Applications of Depth Estimation 
6. Future Research Directions 
7. Conclusion 

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

### 2.1 Paper list

+ **📄 Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation**

   Authors: Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, Bingyi Kang

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2412.14015-b31b1b.svg)](https://arxiv.org/abs/2412.14015)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://promptda.github.io/)
   [![Code](https://img.shields.io/github/stars/DepthAnything/PromptDA.svg?style=social&label=Star)](https://github.com/DepthAnything/PromptDA)

   <details>
   <summary>Click to view Abstract</summary>

   Prompts play a critical role in unleashing the power of language and vision foundation models for specific tasks. For the first time, we introduce prompting into depth foundation models, creating a new paradigm for metric depth estimation termed Prompt Depth Anything. Specifically, we use a low-cost LiDAR as the prompt to guide the Depth Anything model for accurate metric depth output, achieving up to 4K resolution. Our approach centers on a concise prompt fusion design that integrates the LiDAR at multiple scales within the depth decoder. To address training challenges posed by limited datasets containing both LiDAR depth and precise GT depth, we propose a scalable data pipeline that includes synthetic data LiDAR simulation and real data pseudo GT depth generation. Our approach sets new state-of-the-arts on the ARKitScenes and ScanNet++ datasets and benefits downstream applications, including 3D reconstruction and generalized robotic grasping.

   </details>

+ **📄 Scalable Autoregressive Monocular Depth Estimation**

  Authors: Jinhong Wang, Jian Liu, Dongqi Tang, Weiqiang Wang, Wentong Li, Danny Chen, Jintai Chen, Jian Wu

  Published: CVPR 2025

  [![Paper](https://img.shields.io/badge/arXiv-2411.11361-b31b1b.svg)](https://arxiv.org/abs/2411.11361)

  <details>
  <summary>Click to view Abstract</summary>

  This paper shows that the autoregressive model is an effective and scalable monocular depth estimator. The idea is simple: tackle the monocular depth estimation (MDE) task with an autoregressive prediction paradigm, based on two core designs. First, the depth autoregressive model (DAR) treats the depth map of different resolutions as a set of tokens, and conducts the low-to-high resolution autoregressive objective with a patch-wise casual mask. Second, DAR recursively discretizes the entire depth range into more compact intervals, and attains the coarse-to-fine granularity autoregressive objective in an ordinal-regression manner. By coupling these two autoregressive objectives, DAR establishes new state-of-the-art (SOTA) on KITTI and NYU Depth v2 by clear margins. Further, the scalable approach allows scaling the model up to 2.0B and achieving the best RMSE of 1.799 on the KITTI dataset (5% improvement) compared to 1.896 by the current SOTA (Depth Anything). DAR further showcases zero-shot generalization ability on unseen datasets. These results suggest that DAR yields superior performance with an autoregressive prediction paradigm, providing a promising approach to equip modern autoregressive large models (e.g., GPT-4o) with depth estimation capabilities.

  </details>

+ **📄 QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge**

   Authors: Xuan Shen, Weize Ma, Jing Liu, Changdi Yang, Rui Ding, Quanyi Wang, Henghui Ding, Wei Niu, Yanzhi Wang, Pu Zhao, Jun Lin, Jiuxiang Gu

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2503.16709-b31b1b.svg)](https://arxiv.org/abs/2503.16709)

   <details>
   <summary>Click to view Abstract</summary>

   Monocular Depth Estimation (MDE) has emerged as a pivotal task in computer vision, supporting numerous real-world applications. However, deploying accurate depth estimation models on resource-limited edge devices, especially Application-Specific Integrated Circuits (ASICs), is challenging due to the high computational and memory demands. Recent advancements in foundational depth estimation deliver impressive results but further amplify the difficulty of deployment on ASICs. To address this, we propose QuartDepth which adopts post-training quantization to quantize MDE models with hardware accelerations for ASICs. Our approach involves quantizing both weights and activations to 4-bit precision, reducing the model size and computation cost. To mitigate the performance degradation, we introduce activation polishing and compensation algorithm applied before and after activation quantization, as well as a weight reconstruction method for minimizing errors in weight quantization. Furthermore, we design a flexible and programmable hardware accelerator by supporting kernel fusion and customized instruction programmability, enhancing throughput and efficiency. Experimental results demonstrate that our framework achieves competitive accuracy while enabling fast inference and higher energy efficiency on ASICs, bridging the gap between high-performance depth estimation and practical edge-device applicability.

   </details>

+ **📄 Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries**

   Authors: Wei Xu, Charles James Wagner, Junjie Luo, Qi Guo

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2503.23606-b31b1b.svg)](https://arxiv.org/abs/2503.23606)

   <details>
   <summary>Click to view Abstract</summary>

   Extracting depth information from photon-limited, defocused images is challenging because depth from defocus (DfD) relies on accurate estimation of defocus blur, which is fundamentally sensitive to image noise. We present a novel approach to robustly measure object depths from photon-limited images along the defocused boundaries. It is based on a new image patch representation, Blurry-Edges, that explicitly stores and visualizes a rich set of low-level patch information, including boundaries, color, and smoothness. We develop a deep neural network architecture that predicts the Blurry-Edges representation from a pair of differently defocused images, from which depth can be calculated using a closed-form DfD relation we derive. The experimental results on synthetic and real data show that our method achieves the highest depth estimation accuracy on photon-limited images compared to a broad range of state-of-the-art DfD methods.

   </details>

+ **📄 OmniStereo: Real-time Omnidirectional Depth Estimation with Multiview Fisheye Cameras**

   Authors: Jiaxi Deng, Yushen Wang, Haitao Meng, Zuoxun Hou, Yi Chang, Gang Chen

   Published: CVPR 2025

   <details>
   <summary>Click to view Abstract</summary>

   Fast and reliable omnidirectional 3D sensing is essential to many applications such as autonomous driving, robotics, and drone navigation. While many well-recognized methods have been developed to produce high-quality omnidirectional 3D information, they are too slow for real-time computation, limiting their feasibility in practical applications. Motivated by these shortcomings, we propose an efficient omnidirectional depth sensing framework, called OmniStereo, which generates high-quality 3D information in real-time. Unlike prior works, OmniStereo employs Cassini projection to simplify the photometric matching and introduces a lightweight stereo matching network to minimize computational overhead. Additionally, OmniStereo proposes a novel fusion method to handle depth discontinuities and invalid pixels, complemented by a refinement module to reduce mapping-introduced errors and recover fine details. As a result, OmniStereo achieves state-of-the-art (SOTA) accuracy, surpassing the second-best method over 32% in MAE, while maintaining real-time efficiency. It operates more than 16.5× faster than the second-best method in accuracy on TITAN RTX, achieving 12.3 FPS on the embedded device Jetson AGX Orin, underscoring its suitability for real-world deployment. The code will be open-sourced upon acceptance of the paper.

   </details>

+ **📄 Video Depth Anything: Consistent Depth Estimation for Super-Long Videos**

   Authors: Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, Bingyi Kang

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2501.12375-b31b1b.svg)](https://arxiv.org/abs/2501.12375)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://videodepthanything.github.io/)
   [![Code](https://img.shields.io/github/stars/DepthAnything/Video-Depth-Anything.svg?style=social&label=Star)](https://github.com/DepthAnything/Video-Depth-Anything)

   <details>
   <summary>Click to view Abstract</summary>

   Depth Anything has achieved remarkable success in monocular depth estimation with strong generalization ability. However, it suffers from temporal inconsistency in videos, hindering its practical applications. Various methods have been proposed to alleviate this issue by leveraging video generation models or introducing priors from optical flow and camera poses. Nonetheless, these methods are only applicable to short videos (< 10 seconds) and require a trade-off between quality and computational efficiency. We propose Video Depth Anything for high-quality, consistent depth estimation in super-long videos (over several minutes) without sacrificing efficiency. We base our model on Depth Anything V2 and replace its head with an efficient spatial-temporal head. We design a straightforward yet effective temporal consistency loss by constraining the temporal depth gradient, eliminating the need for additional geometric priors. The model is trained on a joint dataset of video depth and unlabeled images, similar to Depth Anything V2. Moreover, a novel key-frame-based strategy is developed for long video inference. Experiments show that our model can be applied to arbitrarily long videos without compromising quality, consistency, or generalization ability. Comprehensive evaluations on multiple video benchmarks demonstrate that our approach sets a new state-of-the-art in zero-shot video depth estimation. We offer models of different scales to support a range of scenarios, with our smallest model capable of real-time performance at 30 FPS.

   </details>

+ **📄 Align3R: Aligned Monocular Depth Estimation for Dynamic Videos**

   Authors: Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, Yuan Liu

   Published: CVPR 2025 Highlight

   [![Paper](https://img.shields.io/badge/arXiv-2412.03079-b31b1b.svg)](https://arxiv.org/abs/2412.03079)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://github.com/jiah-cloud/Align3R)
   [![Code](https://img.shields.io/github/stars/jiah-cloud/Align3R.svg?style=social&label=Star)](https://github.com/jiah-cloud/Align3R)

   <details>
   <summary>Click to view Abstract</summary>

   Recent developments in monocular depth estimation methods enable high-quality depth estimation of single-view images but fail to estimate consistent video depth across different frames. Recent works address this problem by applying a video diffusion model to generate video depth conditioned on the input video, which is training-expensive and can only produce scale-invariant depth values without camera poses. In this paper, we propose a novel video-depth estimation method called Align3R to estimate temporally consistent depth maps for a dynamic video. Our key idea is to utilize the recent DUSt3R model to align estimated monocular depth maps of different timesteps. First, we fine-tune the DUSt3R model with additional estimated monocular depth as inputs for the dynamic scenes. Then, we apply optimization to reconstruct both depth maps and camera poses. Extensive experiments demonstrate that Align3R estimates consistent video depth and camera poses for a monocular video with superior performance compared to baseline methods.

   </details>

+ **📄 Vision-Language Embodiment for Monocular Depth Estimation**

   Authors: Jinchang Zhang, Guoyu Lu

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2503.16535-b31b1b.svg)](https://www.arxiv.org/abs/2503.16535)

   <details>
   <summary>Click to view Abstract</summary>

   Depth estimation is a core problem in robotic perception and vision tasks, but 3D reconstruction from a single image presents inherent uncertainties. Current depth estimation models primarily rely on inter-image relationships for supervised training, often overlooking the intrinsic information provided by the camera itself. We propose a method that embodies the camera model and its physical characteristics into a deep learning model, computing embodied scene depth through real-time interactions with road environments. The model can calculate embodied scene depth in real-time based on immediate environmental changes using only the intrinsic properties of the camera, without any additional equipment. By combining embodied scene depth with RGB image features, the model gains a comprehensive perspective on both geometric and visual details. Additionally, we incorporate text descriptions containing environmental content and depth information as priors for scene understanding, enriching the model's perception of objects. This integration of image and language - two inherently ambiguous modalities - leverages their complementary strengths for monocular depth estimation. The real-time nature of the embodied language and depth prior model ensures that the model can continuously adjust its perception and behavior in dynamic environments. Experimental results show that the embodied depth estimation method enhances model performance across different scenes.

   </details>

+ **📄 Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera**

   Authors: Yuliang Guo, Sparsh Garg, S. Mahdi H. Miangoleh, Xinyu Huang, Liu Ren

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2501.02464-b31b1b.svg)](https://arxiv.org/abs/2501.02464)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://yuliangguo.github.io/depth-any-camera/)
   [![Code](https://img.shields.io/github/stars/yuliangguo/depth_any_camera.svg?style=social&label=Star)](https://github.com/yuliangguo/depth_any_camera)

   <details>
   <summary>Click to view Abstract</summary>

   While recent depth foundation models exhibit strong zero-shot generalization, achieving accurate metric depth across diverse camera types-particularly those with large fields of view (FoV) such as fisheye and 360-degree cameras-remains a significant challenge. This paper presents Depth Any Camera (DAC), a powerful zero-shot metric depth estimation framework that extends a perspective-trained model to effectively handle cameras with varying FoVs. The framework is designed to ensure that all existing 3D data can be leveraged, regardless of the specific camera types used in new applications. Remarkably, DAC is trained exclusively on perspective images but generalizes seamlessly to fisheye and 360-degree cameras without the need for specialized training data. DAC employs Equi-Rectangular Projection (ERP) as a unified image representation, enabling consistent processing of images with diverse FoVs. Its core components include pitch-aware Image-to-ERP conversion with efficient online augmentation to simulate distorted ERP patches from undistorted inputs, FoV alignment operations to enable effective training across a wide range of FoVs, and multi-resolution data augmentation to further address resolution disparities between training and testing. DAC achieves state-of-the-art zero-shot metric depth estimation, improving δ1 accuracy by up to 50% on multiple fisheye and 360-degree datasets compared to prior metric depth foundation models, demonstrating robust generalization across camera types.

   </details>

+ **📄 Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation**

   Authors: Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, Bingyi Kang

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2412.14015-b31b1b.svg)](https://arxiv.org/abs/2412.14015)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://promptda.github.io/)
   [![Code](https://img.shields.io/github/stars/DepthAnything/PromptDA.svg?style=social&label=Star)](https://github.com/DepthAnything/PromptDA)

   <details>
   <summary>Click to view Abstract</summary>

   Prompts play a critical role in unleashing the power of language and vision foundation models for specific tasks. For the first time, we introduce prompting into depth foundation models, creating a new paradigm for metric depth estimation termed Prompt Depth Anything. Specifically, we use a low-cost LiDAR as the prompt to guide the Depth Anything model for accurate metric depth output, achieving up to 4K resolution. Our approach centers on a concise prompt fusion design that integrates the LiDAR at multiple scales within the depth decoder. To address training challenges posed by limited datasets containing both LiDAR depth and precise GT depth, we propose a scalable data pipeline that includes synthetic data LiDAR simulation and real data pseudo GT depth generation. Our approach sets new state-of-the-arts on the ARKitScenes and ScanNet++ datasets and benefits downstream applications, including 3D reconstruction and generalized robotic grasping.

   </details>

+ **📄 Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation**

   Authors: Mehdi Zayene, Jannik Endres, Albias Havolli, Charles Corbière, Salim Cherkaoui, Alexandre Kontouli, Alexandre Alahi

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2411.18335-b31b1b.svg)](https://arxiv.org/abs/2411.18335)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://vita-epfl.github.io/Helvipad/)
   [![Code](https://img.shields.io/github/stars/vita-epfl/Helvipad.svg?style=social&label=Star)](https://github.com/vita-epfl/Helvipad)

   <details>
   <summary>Click to view Abstract</summary>

   Despite progress in stereo depth estimation, omnidirectional imaging remains underexplored, mainly due to the lack of appropriate data. We introduce Helvipad, a real-world dataset for omnidirectional stereo depth estimation, featuring 40K video frames from video sequences across diverse environments, including crowded indoor and outdoor scenes with various lighting conditions. Collected using two 360° cameras in a top-bottom setup and a LiDAR sensor, the dataset includes accurate depth and disparity labels by projecting 3D point clouds onto equirectangular images. Additionally, we provide an augmented training set with increased label density by using depth completion. We benchmark leading stereo depth estimation models for both standard and omnidirectional images. The results show that while recent stereo methods perform decently, a challenge persists in accurately estimating depth in omnidirectional imaging. To address this, we introduce necessary adaptations to stereo models, leading to improved performance.

   </details>

+ **📄 Multi-view Reconstruction via SfM-guided Monocular Depth Estimation**

   Authors: Haoyu Guo, He Zhu, Sida Peng, Haotong Lin, Yunzhi Yan, Tao Xie, Wenguan Wang, Xiaowei Zhou, Hujun Bao

   Published: CVPR 2025 Oral

   [![Paper](https://img.shields.io/badge/arXiv-2503.14483-b31b1b.svg)](https://arxiv.org/abs/2503.14483)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://zju3dv.github.io/murre/)
   [![Code](https://img.shields.io/github/stars/zju3dv/Murre.svg?style=social&label=Star)](https://github.com/zju3dv/Murre)

   <details>
   <summary>Click to view Abstract</summary>

   In this paper, we present a new method for multi-view geometric reconstruction. In recent years, large vision models have rapidly developed, performing excellently across various tasks and demonstrating remarkable generalization capabilities. Some works use large vision models for monocular depth estimation, which have been applied to facilitate multi-view reconstruction tasks in an indirect manner. Due to the ambiguity of the monocular depth estimation task, the estimated depth values are usually not accurate enough, limiting their utility in aiding multi-view reconstruction. We propose to incorporate SfM information, a strong multi-view prior, into the depth estimation process, thus enhancing the quality of depth prediction and enabling their direct application in multi-view geometric reconstruction. Experimental results on public real-world datasets show that our method significantly improves the quality of depth estimation compared to previous monocular depth estimation works. Additionally, we evaluate the reconstruction quality of our approach in various types of scenes including indoor, streetscape, and aerial views, surpassing state-of-the-art MVS methods.

   </details>

+ **📄 GeoDepth: From Point-to-Depth to Plane-to-Depth Modeling for Self-Supervised Monocular Depth Estimation**

   Authors: Haifeng Wu, Shuhang Gu, Lixin Duan, Wen Li

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/CVPR-33131-blue.svg)](https://cvpr.thecvf.com/virtual/2025/poster/33131)

   <details>
   <summary>Click to view Abstract</summary>

   Self-supervised monocular depth estimation has long been treated as a point-wise prediction problem, where the depth of each pixel is usually estimated independently. However, artifacts are often observed in the estimated depth map, e.g., depth values for points located in the same region may jump dramatically. To address this issue, we propose a novel self-supervised monocular depth estimation framework called GeoDepth, where we explore the intrinsic geometric representation in 3D scene for producing accurate and continuous depth map. In particular, we model the complex 3D scene as a collection of planes with varying sizes, where each plane is characterized by a unique set of parameters, namely planar normal (indicating plane orientation) and planar offset (defining the perpendicular distance from the camera center to the plane). Under this modeling, points in the same plane are enforced to share a unique representation, and their depth variations are related only to pixel coordinates. Thus, this geometric relationship can be exploited to regularize the depth variations of these points. To this end, we design a structured plane generation module that introduces temporal-spatial geometric cues and the plane uniqueness principle to recover the correct scene plane representation. In addition, we develop a depth discontinuity module to dynamically identify depth discontinuity regions and subsequently optimize them. Our experiments on the KITTI and NYUv2 datasets demonstrate that GeoDepth achieves state-of-the-art performance, with additional tests on Make3D and ScanNet validating its generalization capabilities.

   </details>

+ **📄 Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses**

   Authors: Yongfan Liu, Hyoukjun Kwon

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2411.10013-b31b1b.svg)](https://arxiv.org/abs/2411.10013)

   <details>
   <summary>Click to view Abstract</summary>

   Stereo depth estimation is a fundamental component in augmented reality (AR) applications. Although AR applications require very low latency for their real-time applications, traditional depth estimation models often rely on time-consuming preprocessing steps such as rectification to achieve high accuracy. Also, non-standard ML operator-based algorithms such as cost volume also require significant latency, which is aggravated on compute resource-constrained mobile platforms. Therefore, we develop hardware-friendly alternatives to the costly cost volume and preprocessing and design two new models based on them, MultiHeadDepth and HomoDepth. Our approaches for cost volume is replacing it with a new group-pointwise convolution-based operator and approximation of cosine similarity based on layernorm and dot product. For online stereo rectification (preprocessing), we introduce a homography matrix prediction network with a rectification positional encoding (RPE), which delivers both low latency and robustness to unrectified images, which eliminates the needs for preprocessing. Our MultiHeadDepth, which includes optimized cost volume, provides 11.8-30.3% improvements in accuracy and 22.9-25.2% reduction in latency compared to a state-of-the-art depth estimation model for AR glasses from industry. Our HomoDepth, which includes optimized preprocessing (Homography + RPE) upon MultiHeadDepth, can process unrectified images and reduce the end-to-end latency by 44.5%. We adopt a multitask learning framework to handle misaligned stereo inputs on HomoDepth, which reduces the AbsRel error by 10.0-24.3%. The results demonstrate the efficacy of our approaches in achieving both high model performance with low latency, which makes a step forward toward practical depth estimation on future AR devices.

   </details>

+ **📄 FoundationStereo: Zero-Shot Stereo Matching**

   Authors: Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield

   Published: CVPR 2025 Oral

   [![Paper](https://img.shields.io/badge/arXiv-2501.09898-b31b1b.svg)](https://arxiv.org/abs/2501.09898)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://nvlabs.github.io/FoundationStereo/)
   [![Code](https://img.shields.io/github/stars/NVlabs/FoundationStereo.svg?style=social&label=Star)](https://github.com/NVlabs/FoundationStereo)

   <details>
   <summary>Click to view Abstract</summary>

   Tremendous progress has been made in deep stereo matching to excel on benchmark datasets through per-domain fine-tuning. However, achieving strong zero-shot generalization — a hallmark of foundation models in other computer vision tasks — remains challenging for stereo matching. We introduce FoundationStereo, a foundation model for stereo depth estimation designed to achieve strong zero-shot generalization. To this end, we first construct a large-scale (1M stereo pairs) synthetic training dataset featuring large diversity and high photorealism, followed by an automatic self-curation pipeline to remove ambiguous samples. We then design a number of network architecture components to enhance scalability, including a side-tuning feature backbone that adapts rich monocular priors from vision foundation models to mitigate the sim-to-real gap, and long-range context reasoning for effective cost volume filtering. Together, these components lead to strong robustness and accuracy across domains, establishing a new standard in zero-shot stereo depth estimation.

   </details>

+ **📄 Synthetic-to-Real Self-supervised Robust Depth Estimation via Learning with Motion and Structure Priors**

   Authors: Weilong Yan, Ming Li, Haipeng Li, Shuwei Shao, Robby T. Tan

   Published: CVPR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2503.20211-b31b1b.svg)](https://arxiv.org/abs/2503.20211)

   <details>
   <summary>Click to view Abstract</summary>

   Self-supervised depth estimation from monocular cameras in diverse outdoor conditions, such as daytime, rain, and nighttime, is challenging due to the difficulty of learning universal representations and the severe lack of labeled real-world adverse data. Previous methods either rely on synthetic inputs and pseudo-depth labels or directly apply daytime strategies to adverse conditions, resulting in suboptimal results. In this paper, we present the first synthetic-to-real robust depth estimation framework, incorporating motion and structure priors to capture real-world knowledge effectively. In the synthetic adaptation, we transfer motion-structure knowledge inside cost volumes for better robust representation, using a frozen daytime model to train a depth estimator in synthetic adverse conditions. In the innovative real adaptation, which targets to fix synthetic-real gaps, models trained earlier identify the weather-insensitive regions with a designed consistency-reweighting strategy to emphasize valid pseudo-labels. We introduce a new regularization by gathering explicit depth distribution to constrain the model facing real-world data. Experiments show that our method outperforms the state-of-the-art across diverse conditions in multi-frame and single-frame evaluations. We achieve improvements of 7.5% and 4.3% in AbsRel and RMSE on average for nuScenes and Robotcar datasets (daytime, nighttime, rain). In zero-shot evaluation of DrivingStereo (rain, fog), our method generalizes better than previous ones.

   </details>

+ **📄 Depth Pro: Sharp Monocular Metric Depth in Less Than a Second**

   Authors: Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun

   Published: ICLR 2025

   [![Paper](https://img.shields.io/badge/arXiv-2410.02073-b31b1b.svg)](https://arxiv.org/abs/2410.02073)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://github.com/apple/ml-depth-pro)
   [![Code](https://img.shields.io/github/stars/apple/ml-depth-pro.svg?style=social&label=Star)](https://github.com/apple/ml-depth-pro)

   <details>
   <summary>Click to view Abstract</summary>

   We present a foundation model for zero-shot metric monocular depth estimation.  
   Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image. Extensive experiments analyze specific design choices and demonstrate that Depth Pro outperforms prior work along multiple dimensions. We release code & weights at [https://github.com/apple/ml-depth-pro](https://github.com/apple/ml-depth-pro).

   </details>

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

   Published: CVPR 2025 Highlight

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

+ **📄 WorDepth: Variational Language Prior for Monocular Depth Estimation**

   Authors: Ziyao Zeng, Daniel Wang, Fengyu Yang, Hyoungseob Park, Yangchao Wu, Stefano Soatto, Byung-Woo Hong, Dong Lao, Alex Wong  
   Published: CVPR 2024  

   [![Paper](https://img.shields.io/badge/arXiv-2404.03635-b31b1b.svg)](https://arxiv.org/abs/2404.03635)  
   [![Code](https://img.shields.io/github/stars/Adonis-galaxy/WorDepth.svg?style=social&label=Star)](https://github.com/Adonis-galaxy/WorDepth)  

   <details>
   <summary>Click to view Abstract</summary>

   Three-dimensional (3D) reconstruction from a single image is an ill-posed problem with inherent ambiguities, i.e. scale. Predicting a 3D scene from text description(s) is similarly ill-posed, i.e. spatial arrangements of objects described. We investigate the question of whether two inherently ambiguous modalities can be used in conjunction to produce metric-scaled reconstructions. To test this, we focus on monocular depth estimation, the problem of predicting a dense depth map from a single image, but with an additional text caption describing the scene. To this end, we begin by encoding the text caption as a mean and standard deviation; using a variational framework, we learn the distribution of the plausible metric reconstructions of 3D scenes corresponding to the text captions as a prior. To "select" a specific reconstruction or depth map, we encode the given image through a conditional sampler that samples from the latent space of the variational text encoder, which is then decoded to the output depth map. Our approach is trained alternatingly between the text and image branches: in one optimization step, we predict the mean and standard deviation from the text description and sample from a standard Gaussian, and in the other, we sample using a (image) conditional sampler. Once trained, we directly predict depth from the encoded text using the conditional sampler. We demonstrate our approach on indoor (NYUv2) and outdoor (KITTI) scenarios, where we show that language can consistently improve performance in both.

   </details>

+ **📄 Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation**

   Authors: Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, Konrad Schindler  

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2312.02145-b31b1b.svg)](https://arxiv.org/abs/2312.02145)  
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://marigoldmonodepth.github.io/)  
   [![Code](https://img.shields.io/github/stars/prs-eth/Marigold.svg?style=social&label=Star)](https://github.com/prs-eth/Marigold)  

   <details>  
   <summary>Click to view Abstract</summary>  

   Monocular depth estimation is a fundamental computer vision task. Recovering 3D depth from a single image is geometrically ill-posed and requires scene understanding, so it is not surprising that the rise of deep learning has led to a breakthrough. The impressive progress of monocular depth estimators has mirrored the growth in model capacity, from relatively modest CNNs to large Transformer architectures. Still, monocular depth estimators tend to struggle when presented with images with unfamiliar content and layout, since their knowledge of the visual world is restricted by the data seen during training, and challenged by zero-shot generalization to new domains. This motivates us to explore whether the extensive priors captured in recent generative diffusion models can enable better, more generalizable depth estimation. We introduce Marigold, a method for affine-invariant monocular depth estimation that is derived from Stable Diffusion and retains its rich prior knowledge. The estimator can be fine-tuned in a couple of days on a single GPU using only synthetic training data. It delivers state-of-the-art performance across a wide range of datasets, including over 20% performance gains in specific cases.  

   </details>

+ **📄 PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation**

   Authors: Zhenyu Li, Shariq Farooq Bhat, Peter Wonka  

   Published: CVPR 2024  

   [![Paper](https://img.shields.io/badge/arXiv-2312.02284-b31b1b.svg)](https://arxiv.org/abs/2312.02284)  
   [![Code](https://img.shields.io/github/stars/zhyever/PatchFusion.svg?style=social&label=Star)](https://github.com/zhyever/PatchFusion)  

   <details>  
   <summary>Click to view Abstract</summary>  

   Single image depth estimation is a foundational task in computer vision and generative modeling. However, prevailing depth estimation models grapple with accommodating the increasing resolutions commonplace in today's consumer cameras and devices. Existing high-resolution strategies show promise, but they often face limitations, ranging from error propagation to the loss of high-frequency details. We present PatchFusion, a novel tile-based framework with three key components to improve the current state of the art: (1) A patch-wise fusion network that fuses a globally-consistent coarse prediction with finer, inconsistent tiled predictions via high-level feature guidance, (2) A Global-to-Local (G2L) module that adds vital context to the fusion network, discarding the need for patch selection heuristics, and (3) A Consistency-Aware Training (CAT) and Inference (CAI) approach, emphasizing patch overlap consistency and thereby eradicating the necessity for post-processing. Experiments on UnrealStereo4K, MVS-Synth, and Middleburry 2014 demonstrate that our framework can generate high-resolution depth maps with intricate details. PatchFusion is independent of the base model for depth estimation. Notably, our framework built on top of SOTA ZoeDepth brings improvements for a total of 17.3% and 29.4% in terms of the root mean squared error (RMSE) on UnrealStereo4K and MVS-Synth, respectively.  

   </details>

+ **📄 From-Ground-To-Objects: Coarse-to-Fine Self-supervised Monocular Depth Estimation of Dynamic Objects with Ground Contact Prior**

   Authors: Jaeho Moon, Juan Luis Gonzalez Bello, Byeongjun Kwon, Munchurl Kim

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2312.10118-b31b1b.svg)](https://arxiv.org/abs/2312.10118)
   [![Code](https://img.shields.io/github/stars/KAIST-VICLab/From_Ground_To_Objects.svg?style=social&label=Star)](https://github.com/KAIST-VICLab/From_Ground_To_Objects)

   <details>
   <summary>Click to view Abstract</summary>

   Self-supervised monocular depth estimation (DE) is an approach to learning depth without costly depth ground truths. However, it often struggles with moving objects that violate the static scene assumption during training. To address this issue, we introduce a coarse-to-fine training strategy leveraging the ground contacting prior based on the observation that most moving objects in outdoor scenes contact the ground. In the coarse training stage, we exclude the objects in dynamic classes from the reprojection loss calculation to avoid inaccurate depth learning. To provide precise supervision on the depth of the objects, we present a novel Ground-contacting-prior Disparity Smoothness Loss (GDS-Loss) that encourages a DE network to align the depth of the objects with their ground-contacting points. Subsequently, in the fine training stage, we refine the DE network to learn the detailed depth of the objects from the reprojection loss, while ensuring accurate DE on the moving object regions by employing our regularization loss with a cost-volume-based weighting factor. Our overall coarse-to-fine training strategy can easily be integrated with existing DE methods without any modifications, significantly enhancing DE performance on challenging Cityscapes and KITTI datasets, especially in the moving object regions.

   </details>

+ **📄 ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation**

   Authors: Suraj Patni, Aradhye Agarwal, Chetan Arora

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2403.18807-b31b1b.svg)](https://arxiv.org/abs/2403.18807)
   [![Code](https://img.shields.io/github/stars/Aradhye2002/EcoDepth.svg?style=social&label=Star)](https://github.com/Aradhye2002/EcoDepth)

   <details>
   <summary>Click to view Abstract</summary>

   In the absence of parallax cues, a learning-based single image depth estimation (SIDE) model relies heavily on shading and contextual cues in the image. While this simplicity is attractive, it is necessary to train such models on large and varied datasets, which are difficult to capture. It has been shown that using embeddings from pre-trained foundational models, such as CLIP, improves zero shot transfer in several applications. Taking inspiration from this, in our paper we explore the use of global image priors generated from a pre-trained ViT model to provide more detailed contextual information. We argue that the embedding vector from a ViT model, pre-trained on a large dataset, captures greater relevant information for SIDE than the usual route of generating pseudo image captions, followed by CLIP based text embeddings. Based on this idea, we propose a new SIDE model using a diffusion backbone which is conditioned on ViT embeddings. Our proposed design establishes a new state-of-the-art (SOTA) for SIDE on NYUv2 dataset, achieving Abs Rel error of 0.059 (14% improvement) compared to 0.069 by the current SOTA (VPD). And on KITTI dataset, achieving Sq Rel error of 0.139 (2% improvement) compared to 0.142 by the current SOTA (GEDepth). For zero-shot transfer with a model trained on NYUv2, we report mean relative improvement of (20%, 23%, 81%, 25%) over NeWCRFs on (Sun-RGBD, iBims1, DIODE, HyperSim) datasets, compared to (16%, 18%, 45%, 9%) by ZoeDepth.

   </details>

+ **📄 Mining Supervision for Dynamic Regions in Self-Supervised Monocular Depth Estimation**

   Authors: Hoang Chuong Nguyen, Tianyu Wang, Jose M. Alvarez, Miaomiao Liu

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2404.14908-b31b1b.svg)](https://arxiv.org/abs/2404.14908)

   <details>
   <summary>Click to view Abstract</summary>

   This paper focuses on self-supervised monocular depth estimation in dynamic scenes trained on monocular videos. Existing methods jointly estimate pixel-wise depth and motion, relying mainly on an image reconstruction loss. Dynamic regions remain a critical challenge for these methods due to the inherent ambiguity in depth and motion estimation, resulting in inaccurate depth estimation. This paper proposes a self-supervised training framework exploiting pseudo depth labels for dynamic regions from training data. The key contribution of our framework is to decouple depth estimation for static and dynamic regions of images in the training data. We start with an unsupervised depth estimation approach, which provides reliable depth estimates for static regions and motion cues for dynamic regions and allows us to extract moving object information at the instance level. In the next stage, we use an object network to estimate the depth of those moving objects assuming rigid motions. Then, we propose a new scale alignment module to address the scale ambiguity between estimated depths for static and dynamic regions. We can then use the depth labels generated to train an end-to-end depth estimation network and improve its performance. Extensive experiments on the Cityscapes and KITTI datasets show that our self-training strategy consistently outperforms existing self/unsupervised depth estimation methods.

   </details>

+ **📄 Mind The Edge: Refining Depth Edges in Sparsely-Supervised Monocular Depth Estimation**

   Authors: Lior Talker, Aviad Cohen, Erez Yosef, Alexandra Dana, Michael Dinerstein

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2212.05315-b31b1b.svg)](https://arxiv.org/abs/2212.05315)
   [![Code](https://img.shields.io/github/stars/liortalker/MindTheEdge.svg?style=social&label=Star)](https://github.com/liortalker/MindTheEdge)

   <details>
   <summary>Click to view Abstract</summary>

   Monocular Depth Estimation (MDE) is a fundamental problem in computer vision with numerous applications. Recently, LIDAR-supervised methods have achieved remarkable per-pixel depth accuracy in outdoor scenes. However, significant errors are typically found in the proximity of depth discontinuities, i.e., depth edges, which often hinder the performance of depth-dependent applications that are sensitive to such inaccuracies, e.g., novel view synthesis and augmented reality. Since direct supervision for the location of depth edges is typically unavailable in sparse LIDAR-based scenes, encouraging the MDE model to produce correct depth edges is not straightforward. To the best of our knowledge this paper is the first attempt to address the depth edges issue for LIDAR-supervised scenes. In this work we propose to learn to detect the location of depth edges from densely-supervised synthetic data, and use it to generate supervision for the depth edges in the MDE training. To quantitatively evaluate our approach, and due to the lack of depth edges GT in LIDAR-based scenes, we manually annotated subsets of the KITTI and the DDAD datasets with depth edges ground truth. We demonstrate significant gains in the accuracy of the depth edges with comparable per-pixel depth accuracy on several challenging datasets.

   </details>

+ **📄 Elite360D: Towards Efficient 360 Depth Estimation via Semantic- and Distance-Aware Bi-Projection Fusion**

   Authors: Hao Ai, Lin Wang

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2403.16376-b31b1b.svg)](https://arxiv.org/abs/2403.16376)

   <details>
   <summary>Click to view Abstract</summary>

   360 depth estimation has recently received great attention for 3D reconstruction owing to its omnidirectional field of view (FoV). Recent approaches are predominantly focused on cross-projection fusion with geometry-based re-projection: they fuse 360 images with equirectangular projection (ERP) and another projection type, e.g., cubemap projection to estimate depth with the ERP format. However, these methods suffer from 1) limited local receptive fields, making it hardly possible to capture large FoV scenes, and 2) prohibitive computational cost, caused by the complex cross-projection fusion module design. In this paper, we propose Elite360D, a novel framework that inputs the ERP image and icosahedron projection (ICOSAP) point set, which is undistorted and spatially continuous. Elite360D is superior in its capacity in learning a representation from a local-with-global perspective. With a flexible ERP image encoder, it includes an ICOSAP point encoder, and a Bi-projection Bi-attention Fusion (B2F) module (totally ~1M parameters). Specifically, the ERP image encoder can take various perspective image-trained backbones (e.g., ResNet, Transformer) to extract local features. The point encoder extracts the global features from the ICOSAP. Then, the B2F module captures the semantic- and distance-aware dependencies between each pixel of the ERP feature and the entire ICOSAP feature set. Without specific backbone design and obvious computational cost increase, Elite360D outperforms the prior arts on several benchmark datasets.

   </details>

+ **📄 Atlantis: Enabling Underwater Depth Estimation with Stable Diffusion**

   Authors: Fan Zhang, Shaodi You, Yu Li, Ying Fu

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2312.12471-b31b1b.svg)](https://arxiv.org/abs/2312.12471)
   [![Code](https://img.shields.io/github/stars/zkawfanx/Atlantis.svg?style=social&label=Star)](https://github.com/zkawfanx/Atlantis)

   <details>
   <summary>Click to view Abstract</summary>

   Monocular depth estimation has experienced significant progress on terrestrial images in recent years, largely due to deep learning advancements. However, it remains inadequate for underwater scenes, primarily because of data scarcity. Given the inherent challenges of light attenuation and backscattering in water, acquiring clear underwater images or precise depth information is notably difficult and costly. Consequently, learning-based approaches often rely on synthetic data or turn to unsupervised or self-supervised methods to mitigate this lack of data. Nonetheless, the performance of these methods is often constrained by the domain gap and looser constraints. In this paper, we propose a novel pipeline for generating photorealistic underwater images using accurate terrestrial depth data. This approach facilitates the training of supervised models for underwater depth estimation, effectively reducing the performance disparity between terrestrial and underwater environments. Contrary to prior synthetic datasets that merely apply style transfer to terrestrial images without altering the scene content, our approach uniquely creates vibrant, non-existent underwater scenes by leveraging terrestrial depth data through the innovative Stable Diffusion model. Specifically, we introduce a unique Depth2Underwater ControlNet, trained on specially prepared \{Underwater, Depth, Text\} data triplets, for this generation task. Our newly developed dataset enables terrestrial depth estimation models to achieve considerable improvements, both quantitatively and qualitatively, on unseen underwater images, surpassing their terrestrial pre-trained counterparts. Moreover, the enhanced depth accuracy for underwater scenes also aids underwater image restoration techniques that rely on depth maps, further demonstrating our dataset's utility.

   </details>

+ **📄 Cross-spectral Gated-RGB Stereo Depth Estimation**

   Authors: Samuel Brucker, Stefanie Walz, Mario Bijelic, Felix Heide

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2405.12759-b31b1b.svg)](https://arxiv.org/abs/2405.12759)

   <details>
   <summary>Click to view Abstract</summary>

   Gated cameras flood-illuminate a scene and capture the time-gated impulse response of a scene. By employing nanosecond-scale gates, existing sensors are capable of capturing mega-pixel gated images, delivering dense depth improving on today's LiDAR sensors in spatial resolution and depth precision. Although gated depth estimation methods deliver a million of depth estimates per frame, their resolution is still an order below existing RGB imaging methods. In this work, we combine high-resolution stereo HDR RCCB cameras with gated imaging, allowing us to exploit depth cues from active gating, multi-view RGB and multi-view NIR sensing -- multi-view and gated cues across the entire spectrum. The resulting capture system consists only of low-cost CMOS sensors and flood-illumination. We propose a novel stereo-depth estimation method that is capable of exploiting these multi-modal multi-view depth cues, including the active illumination that is measured by the RCCB camera when removing the IR-cut filter. The proposed method achieves accurate depth at long ranges, outperforming the next best existing method by 39% for ranges of 100 to 220m in MAE on accumulated LiDAR ground-truth.

   </details>

+ **📄 Depth Prompting for Sensor-Agnostic Depth Estimation**

   Authors: Jin-Hwi Park, Chanhwi Jeong, Junoh Lee, Hae-Gon Jeon

   Published: CVPR 2024

   [![Paper](https://img.shields.io/badge/arXiv-2405.11867-b31b1b.svg)](https://arxiv.org/abs/2405.11867)
   [![Code](https://img.shields.io/github/stars/JinhwiPark/DepthPrompting.svg?style=social&label=Star)](https://github.com/JinhwiPark/DepthPrompting)

   <details>
   <summary>Click to view Abstract</summary>

   Dense depth maps have been used as a key element of visual perception tasks. There have been tremendous efforts to enhance the depth quality, ranging from optimization-based to learning-based methods. Despite the remarkable progress for a long time, their applicability in the real world is limited due to systematic measurement biases such as density, sensing pattern, and scan range. It is well-known that the biases make it difficult for these methods to achieve their generalization. We observe that learning a joint representation for input modalities (e.g., images and depth), which most recent methods adopt, is sensitive to the biases. In this work, we disentangle those modalities to mitigate the biases with prompt engineering. For this, we design a novel depth prompt module to allow the desirable feature representation according to new depth distributions from either sensor types or scene configurations. Our depth prompt can be embedded into foundation models for monocular depth estimation. Through this embedding process, our method helps the pretrained model to be free from restraint of depth scan range and to provide absolute scale depth maps. We demonstrate the effectiveness of our method through extensive evaluations.

   </details>

+ **📄 SE(3) Equivariant Ray Embeddings for Implicit Multi-View Depth Estimation**

   Authors: Yinshuang Xu, Dian Chen, Katherine Liu, Sergey Zakharov, Rares Ambrus, Kostas Daniilidis, Vitor Guizilini

   Published: NeurIPS 2024

   [![Paper](https://img.shields.io/badge/arXiv-2411.07326-b31b1b.svg)](https://arxiv.org/abs/2411.07326)

   <details>
   <summary>Click to view Abstract</summary>

   Incorporating inductive bias by embedding geometric entities (such as rays) as input has proven successful in multi-view learning. However, the methods adopting this technique typically lack equivariance, which is crucial for effective 3D learning. Equivariance serves as a valuable inductive prior, aiding in the generation of robust multi-view features for 3D scene understanding. In this paper, we explore the application of equivariant multi-view learning to depth estimation, not only recognizing its significance for computer vision and robotics but also addressing the limitations of previous research. Most prior studies have either overlooked equivariance in this setting or achieved only approximate equivariance through data augmentation, which often leads to inconsistencies across different reference frames. To address this issue, we propose to embed SE(3) equivariance into the Perceiver IO architecture. We employ Spherical Harmonics for positional encoding to ensure 3D rotation equivariance, and develop a specialized equivariant encoder and decoder within the Perceiver IO architecture. To validate our model, we applied it to the task of stereo depth estimation, achieving state of the art results on real-world datasets without explicit geometric constraints or extensive data augmentation.

   </details>

+ **📄 BetterDepth: Plug-and-Play Diffusion Refiner for Zero-Shot Monocular Depth Estimation**

   Authors: Xiang Zhang, Bingxin Ke, Hayko Riemenschneider, Nando Metzger, Anton Obukhov, Markus Gross, Konrad Schindler, Christopher Schroers

   Published: NeurIPS 2024

   [![Paper](https://img.shields.io/badge/arXiv-2407.17952-b31b1b.svg)](https://arxiv.org/abs/2407.17952)

   <details>
   <summary>Click to view Abstract</summary>

   By training over large-scale datasets, zero-shot monocular depth estimation (MDE) methods show robust performance in the wild but often suffer from insufficient detail. Although recent diffusion-based MDE approaches exhibit a superior ability to extract details, they struggle in geometrically complex scenes that challenge their geometry prior, trained on less diverse 3D data. To leverage the complementary merits of both worlds, we propose BetterDepth to achieve geometrically correct affine-invariant MDE while capturing fine details. Specifically, BetterDepth is a conditional diffusion-based refiner that takes the prediction from pre-trained MDE models as depth conditioning, in which the global depth layout is well-captured, and iteratively refines details based on the input image. For the training of such a refiner, we propose global pre-alignment and local patch masking methods to ensure BetterDepth remains faithful to the depth conditioning while learning to add fine-grained scene details. With efficient training on small-scale synthetic datasets, BetterDepth achieves state-of-the-art zero-shot MDE performance on diverse public datasets and on in-the-wild scenes. Moreover, BetterDepth can improve the performance of other MDE models in a plug-and-play manner without further re-training.

   </details>

+ **📄 Depth Anything V2**

   Authors: Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao

   Published: NeurIPS 2024

   [![Paper](https://img.shields.io/badge/arXiv-2406.09414-b31b1b.svg)](https://arxiv.org/abs/2406.09414)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://depth-anything-v2.github.io/)
   [![Code](https://img.shields.io/github/stars/DepthAnything/Depth-Anything-V2.svg?style=social&label=Star)](https://github.com/DepthAnything/Depth-Anything-V2)

   <details>
   <summary>Click to view Abstract</summary>

   This work presents Depth Anything V2. Without pursuing fancy techniques, we aim to reveal crucial findings to pave the way towards building a powerful monocular depth estimation model. Notably, compared with V1, this version produces much finer and more robust depth predictions through three key practices: 1) replacing all labeled real images with synthetic images, 2) scaling up the capacity of our teacher model, and 3) teaching student models via the bridge of large-scale pseudo-labeled real images. Compared with the latest models built on Stable Diffusion, our models are significantly more efficient (more than 10x faster) and more accurate. We offer models of different scales (ranging from 25M to 1.3B params) to support extensive scenarios. Benefiting from their strong generalization capability, we fine-tune them with metric depth labels to obtain our metric depth models. In addition to our models, considering the limited diversity and frequent noise in current test sets, we construct a versatile evaluation benchmark with precise annotations and diverse scenes to facilitate future research.

   </details>

+ **📄 Depth Anywhere: Enhancing 360 Monocular Depth Estimation via Perspective Distillation and Unlabeled Data Augmentation**

   Authors: Ning-Hsu Wang, Yu-Lun Liu

   Published: NeurIPS 2024

   [![Paper](https://img.shields.io/badge/arXiv-2406.12849-b31b1b.svg)](https://arxiv.org/abs/2406.12849)
   [![Project](https://img.shields.io/badge/Project-Page-blue)](https://albert100121.github.io/Depth-Anywhere/)
   [![Code](https://img.shields.io/github/stars/albert100121/Depth-Anywhere.svg?style=social&label=Star)](https://github.com/albert100121/Depth-Anywhere)

   <details>
   <summary>Click to view Abstract</summary>

   Accurately estimating depth in 360-degree imagery is crucial for virtual reality, autonomous navigation, and immersive media applications. Existing depth estimation methods designed for perspective-view imagery fail when applied to 360-degree images due to different camera projections and distortions, whereas 360-degree methods perform inferior due to the lack of labeled data pairs. We propose a new depth estimation framework that utilizes unlabeled 360-degree data effectively. Our approach uses state-of-the-art perspective depth estimation models as teacher models to generate pseudo labels through a six-face cube projection technique, enabling efficient labeling of depth in 360-degree images. This method leverages the increasing availability of large datasets. Our approach includes two main stages: offline mask generation for invalid regions and an online semi-supervised joint training regime. We tested our approach on benchmark datasets such as Matterport3D and Stanford2D3D, showing significant improvements in depth estimation accuracy, particularly in zero-shot scenarios. Our proposed training pipeline can enhance any 360 monocular depth estimator and demonstrates effective knowledge transfer across different camera projections and data types.

   </details>

+ **📄 Metric from Human: Zero-shot Monocular Metric Depth Estimation via Test-time Adaptation**

   Authors: Yizhou Zhao, Hengwei Bian, Kaihua Chen, Pengliang Ji, Liao Qu, Shao-yu Lin, Weichen Yu, Haoran Li, Hao Chen, Jun Shen, Bhiksha Raj, Min Xu

   Published: NeurIPS 2024

   [![Code](https://img.shields.io/github/stars/Skaldak/MfH.svg?style=social&label=Star)](https://github.com/Skaldak/MfH)

   <details>
   <summary>Click to view Abstract</summary>

   Monocular depth estimation (MDE) is fundamental for deriving 3D scene structures from 2D images. While state-of-the-art monocular relative depth estimation (MRDE) excels in estimating relative depths for in-the-wild images, current monocular metric depth estimation (MMDE) approaches still face challenges in handling unseen scenes. Since MMDE can be viewed as the composition of MRDE and metric scale recovery, we attribute this difficulty to scene dependency, where MMDE models rely on scenes observed during supervised training for predicting scene scales during inference. To address this issue, we propose to use humans as landmarks for distilling scene-independent metric scale priors from generative painting models. Our approach, Metric from Human (MfH), bridges from generalizable MRDE to zero-shot MMDE in a generate-and-estimate manner. Specifically, MfH generates humans on the input image with generative painting and estimates human dimensions with an off-the-shelf human mesh recovery (HMR) model. Based on MRDE predictions, it propagates the metric information from painted humans to the contexts, resulting in metric depth estimations for the original input. Through this annotation-free test-time adaptation, MfH achieves superior zero-shot performance in MMDE, demonstrating its strong generalization ability.

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

   [![Paper](https://img.shields.io/badge/arXiv-2312.14132-b31b1b.svg)](https://arxiv.org/pdf/2312.14132)
   [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)
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


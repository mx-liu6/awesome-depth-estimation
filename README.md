# Awesome-Depth-Estimation
A curated list of papers and resources focused on Depth Estimation. 

## Selected Papers on Depth Estimation

### **UniDepth: Universal Monocular Metric Depth Estimation**  

Authors: Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu  

Published: CVPR 2024

**[Paper](https://arxiv.org/pdf/2403.18913)** | **[Project](https://lpiccinelli-eth.github.io/pub/unidepth/)** | **[Code](https://github.com/lpiccinelli-eth/UniDepth)**

**Keywords**: *Monocular Depth Estimation*, *3D Perception*, *Zero-shot Learning*, *Geometric Invariance Loss*

<details>
  <summary>Click to view Abstract</summary>

  Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepth, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE methods, UniDepth directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepth implements a self-promptable camera module predicting dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. Thorough evaluations on ten datasets in a zero-shot regime consistently demonstrate the superior performance of UniDepth, even when compared with methods directly trained on the testing domains.

</details>

---


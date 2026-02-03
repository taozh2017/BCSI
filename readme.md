# [AAAI 2026] BCSI: Bidirectional Channel-selective Semantic Interaction for Semi-supervised Medical Image Segmentation

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of the paper: **"Bidirectional Channel-selective Semantic Interaction (BCSI) framework for semi-supervised medical image segmentation"**, accepted by **AAAI 2026**.

## 📖 Framework

![BCSI Framework](assets/Model.png)

We propose **BCSI**, a novel framework that introduces:
1.  **Weak-to-Strong Consistency:** Utilizing Semantic-Spatial Perturbation (SSP) to mitigate error propagation.
2.  **Channel-selective Router (CR):** Dynamically selecting critical channels to filter noise.
3.  **Bidirectional Interaction (BCI):** facilitating deep feature exchange between data streams.

Experimental results on **LA, Pancreas-CT, and BraTS-2019** datasets demonstrate that BCSI significantly outperforms state-of-the-art methods.



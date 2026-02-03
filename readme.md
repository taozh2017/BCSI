# [AAAI 2026] BCSI: Bidirectional Channel-selective Semantic Interaction for Semi-supervised Medical Image Segmentation

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official implementation of the paper: **"Bidirectional Channel-selective Semantic Interaction (BCSI) framework for semi-supervised medical image segmentation"**, accepted by **AAAI 2026**.

## 📂 Dataset Preprocessing

We provide the pre-processed [**BraTS 2019**](https://drive.google.com/file/d/1fqev6O8Urq2pY14fIFyoiFEROj3HDK_p/view?usp=sharing) dataset used in our paper.

## Training
To train the BCSI framework with different labeled data ratios (e.g., 10% labels), use the following commands:

```bash
python train.py --data_path /your/local/path/to/BraTS2019 --dataset BraTS2019 --labeled_num 10
```

## Run Inference
To test the trained BCSI model on the BraTS 2019 test set, run the following command:

```bash
python prediction.py --data_path /your/local/path/to/BraTS2019 --dataset BraTS2019 --model_path ./Results/..pth
```

## 📖 Framework

![BCSI Framework](assets/Model.png)

We propose **BCSI**, a novel framework that introduces:
1.  **Weak-to-Strong Consistency:** Utilizing Semantic-Spatial Perturbation (SSP) to mitigate error propagation.
2.  **Channel-selective Router (CR):** Dynamically selecting critical channels to filter noise.
3.  **Bidirectional Interaction (BCI):** facilitating deep feature exchange between data streams.

Experimental results on **LA, Pancreas-CT, and BraTS-2019** datasets demonstrate that BCSI significantly outperforms state-of-the-art methods.



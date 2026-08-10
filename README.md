# HiSem: Hierarchical Semantic Disentangling for Remote Sensing Image Change Captioning

<p align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/11570222">
    <img src="https://img.shields.io/badge/Paper-IEEE-blue" alt="Paper">
  </a>
  <a href="https://scholar.googleusercontent.com/scholar.bib?q=info:1xNgkOZOfdkJ:scholar.google.com/&output=citation&scisdr=CskwzKogEOOEvKpPchM:AM1tuoMAAAAAanlJahNafmbh4Dvu5FGxoHM0tTg&scisig=AM1tuoMAAAAAanlJav4aCTx2gGuFwkHDJ26LJVI&scisf=4&ct=citation&cd=-1&hl=zh-CN">
    <img src="https://img.shields.io/badge/Citation-BibTeX-lightgrey" alt="Citation">
  </a>
</p>

<p align="center">
  <strong>Accepted by IEEE TGRS 2026</strong>
</p>

<p align="center">
  <a href="https://man-wang-star.github.io/">Man Wang</a>,
  <a href="https://chen-yang-liu.github.io/">Chenyang Liu</a>,
  <strong>Wenjun Li</strong>,
  <strong>Feng Ni</strong>,
  <a href="https://ccs.imu.edu.cn/info/1020/2043.htm">Bing Jia</a>,
  <a href="https://ccs.imu.edu.cn/info/1020/1847.htm">Baoqi Huang</a>,
  <a href="https://ccs.imu.edu.cn/info/1023/2411.htm">Riting Xia</a>,
  <a href="https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ">Zhenwei Shi</a>
</p>

---

## ⭐ Introduction

This repository provides the official implementation of:

> **[Full Paper Title]**

[用一至两段英文介绍论文的研究问题、核心方法和主要贡献。]

The main contributions of this work include:

- **[Contribution 1]**：介绍第一个主要创新点。
- **[Contribution 2]**：介绍第二个主要创新点。
- **[Contribution 3]**：介绍实验结果或性能优势。

If this project is helpful to your research, please consider giving it a ⭐.

---

## 🔥 News

- **[YYYY.MM]** — The source code has been released.
- **[YYYY.MM]** — The paper has been accepted by **[Journal or Conference]**.
- **[YYYY.MM]** — The pretrained models have been released.
- **[YYYY.MM]** — The training and evaluation instructions have been updated.

---

## 📢 Overview

<p align="center">
  <img src="assets/framework.png" width="95%" alt="Framework">
</p>

<p align="center">
  <em>Overview of the proposed [Project Name] framework.</em>
</p>

[在这里简要解释框架图，例如输入、特征提取模块、核心创新模块和最终输出。]

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/[USERNAME]/[REPOSITORY].git
cd [REPOSITORY]
```

### 2. Create the environment

```bash
conda create -n [ENV_NAME] python=3.9
conda activate [ENV_NAME]
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install additional packages

```bash
pip install [PACKAGE_NAME]
```

> **Note:** [填写需要特别说明的 CUDA、PyTorch、操作系统或第三方库版本。]

---

## 📂 Data Preparation

### Dataset

Download the dataset from the following link:

- [Dataset Name] — [Download Link]
- [Another Dataset] — [Download Link]

Organize the dataset as follows:

```text
dataset/
├── train/
│   ├── images_A/
│   ├── images_B/
│   └── annotations.json
├── val/
│   ├── images_A/
│   ├── images_B/
│   └── annotations.json
└── test/
    ├── images_A/
    ├── images_B/
    └── annotations.json
```

### Data preprocessing

```bash
python tools/preprocess.py \
    --input_path /path/to/raw/dataset \
    --output_path /path/to/processed/dataset
```

> **Note:** [填写预处理时需要注意的参数、路径或数据划分方式。]

---

## 🚀 Training

Run the following command to train the model:

```bash
python train.py \
    --data_root /path/to/dataset \
    --config configs/[CONFIG_FILE].yaml \
    --output_dir outputs/[EXPERIMENT_NAME]
```

For multi-GPU training:

```bash
torchrun \
    --nproc_per_node=[GPU_NUMBER] \
    train.py \
    --data_root /path/to/dataset \
    --config configs/[CONFIG_FILE].yaml
```

---

## 🔍 Evaluation

Evaluate a trained model using:

```bash
python test.py \
    --data_root /path/to/dataset \
    --checkpoint /path/to/checkpoint.pth
```

The evaluation results will be saved in:

```text
outputs/evaluation/
```

---

## 📦 Pretrained Models

| Model | Dataset | Parameters | Performance | Download |
|:------|:--------|-----------:|------------:|:--------:|
| [Model A] | [Dataset A] | [XX M] | [Score] | [Link] |
| [Model B] | [Dataset B] | [XX M] | [Score] | [Link] |

> The pretrained checkpoints are provided for academic research purposes.

---

## 📊 Experimental Results

### Quantitative comparison

| Method | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|:-------|-------:|-------:|--------:|------:|
| Baseline A | 0.00 | 0.00 | 0.00 | 0.00 |
| Baseline B | 0.00 | 0.00 | 0.00 | 0.00 |
| **Ours** | **0.00** | **0.00** | **0.00** | **0.00** |

### Qualitative results

<p align="center">
  <img src="assets/results.png" width="95%" alt="Qualitative Results">
</p>

<p align="center">
  <em>Qualitative comparison between the proposed method and existing approaches.</em>
</p>

---

## 📁 Repository Structure

```text
[REPOSITORY]/
├── configs/              # Configuration files
├── datasets/             # Dataset loading and preprocessing
├── models/               # Model definitions
├── tools/                # Utility scripts
├── assets/               # README images
├── train.py              # Training script
├── test.py               # Evaluation script
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚠️ Notes

- The code has been tested with **Python [VERSION]** and **PyTorch [VERSION]**.
- The recommended CUDA version is **CUDA [VERSION]**.
- Please update dataset and checkpoint paths before running the scripts.
- [填写其他可能影响复现结果的重要事项。]

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@article{[CITATION_KEY],
  title   = {[PAPER_TITLE]},
  author  = {[AUTHOR_LIST]},
  journal = {[JOURNAL_NAME]},
  year    = {[YEAR]},
  volume  = {[VOLUME]},
  number  = {[NUMBER]},
  pages   = {[PAGES]},
  doi     = {[DOI]}
}
```

---

## 🤝 Acknowledgements

This project is inspired by or built upon the following repositories:

- [Project A](https://github.com/[OWNER]/[PROJECT_A])
- [Project B](https://github.com/[OWNER]/[PROJECT_B])
- [Project C](https://github.com/[OWNER]/[PROJECT_C])

We sincerely thank the authors for making their work publicly available.

---

## 📬 Contact

For questions or suggestions, please contact:

- **Name:** [Your Name]
- **Email:** [your-email@example.com]
- **Homepage:** [https://your-homepage.github.io/](https://your-homepage.github.io/)
- **GitHub:** [https://github.com/USERNAME](https://github.com/USERNAME)

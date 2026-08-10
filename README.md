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
  <a href="https://man-wang-star.github.io/">Man Wang</a>,
  <a href="https://chen-yang-liu.github.io/">Chenyang Liu</a>,
  <strong>Wenjun Li</strong>,
  <strong>Feng Ni</strong>,
  <a href="https://ccs.imu.edu.cn/info/1020/2043.htm">Bing Jia</a>,
  <a href="https://ccs.imu.edu.cn/info/1020/1847.htm">Baoqi Huang</a>,
  <a href="https://ccs.imu.edu.cn/info/1023/2411.htm">Riting Xia</a>,
  <a href="https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ">Zhenwei Shi</a>
</p>

<p align="center">
  <img src="resource/framework.png" width="95%" alt="Framework">
</p>

<p align="center">
  <strong>Accepted by IEEE TGRS 2026 🎉🎉🎉</strong>
</p>

---

## ⭐ Introduction

This repository provides the official implementation of:

> **HiSem: Hierarchical Semantic Disentangling for Remote Sensing Image Change Captioning**

If this project is helpful to your research, please consider giving it a ⭐.

---

## 🛠️ Installation and Dependencies

```bash
git clone https://github.com/Man-Wang-star/HiSem.git
cd HiSem
conda create -n HiSem_env python=3.9
conda activate HiSem_env
pip install -r requirements.txt
```

---

## 📂 Data Preparation

### Dataset

Download the dataset from the following link:

- <a href="https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset">LEVIR_CC dataset</a>
- <a href="https://www.kaggle.com/datasets/yuehaozhang1109/whu-cdc">WHU_CDC dataset</a>

The LEVIR-CC and WHU-CDC datasets share the same data structure, organized as follows:

```text
├─/root/Data/LEVIR_CC/
        ├─LevirCCcaptions.json
        ├─images
             ├─train
             │  ├─A
             │  ├─B
             ├─val
             │  ├─A
             │  ├─B
             ├─test
             │  ├─A
             │  ├─B
```

### Data preprocessing

```bash
python preprocess_data.py
```

---

## 🚀 Training

Run the following command to train the model:

```bash
python train.py
```

---

## 🔍 Evaluation

Evaluate a trained model using:

```bash
python test.py
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

<p align="center">
  <img src="resource/ablation.png" width="95%" alt="Qualitative Results">
</p>

### Qualitative results

<p align="center">
  <img src="resource/comparison_LEVIR-CC.png" width="95%" alt="Qualitative Results">
</p>

<p align="center">
  <img src="resource/comparison_WHU-CDC.png" width="95%" alt="Qualitative Results">
</p>

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

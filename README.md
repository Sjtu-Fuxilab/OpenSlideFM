# OpenSlideFM

A computationally efficient multi-scale foundation model for computational pathology.

**71M parameters · Single consumer GPU · 2.3 sec/WSI**

## Overview

OpenSlideFM processes whole-slide images at two resolutions (0.5 and 2.0 μm/pixel) to capture both cellular morphology and tissue architecture. Pre-trained with BYOL + Masked Feature Reconstruction on 20,000 TCGA slides spanning 31 cancer types.

| Component | Detail |
|-----------|--------|
| Backbone | ConvNeXt-Tiny (28M, 768-d) |
| Aggregator | 6-layer Transformer (43M) |
| Token budget | 1,600 (1,200 high-res + 400 low-res) |
| Pre-training | BYOL (EMA=0.996) + MFR (25% mask) |
| Hardware | NVIDIA RTX 4090 (24 GB), 72 hrs |

## Results

| Task | Dataset | Metric | Result |
|------|---------|--------|--------|
| Pan-cancer classification (31-class) | TCGA (20K slides) | Accuracy | 81.21% |
| 10-class benchmark | TCGA (4,044 patients) | Accuracy | 91.0% ± 2.6% |
| Lymph node metastasis detection | CAMELYON16 (269 slides) | AUROC | 0.673 |
| Multi-center pN staging | CAMELYON17 (499 slides) | Quadratic κ | 0.141 |
| Prostate cancer grading | PANDA (10,616 slides) | Quadratic κ | 0.826 |


## Pipeline

`notebook.py` contains the full training and evaluation pipeline (Scripts 01–12):

```
WSI Input → Tissue Segmentation → Two-Scale Tiling (0.5 + 2.0 μm/pixel)
         → ConvNeXt-Tiny Feature Extraction → Transformer Aggregation
         → BYOL + MFR Pre-training → Downstream Evaluation
```

## Data

- **TCGA**: [NIH Genomic Data Commons](https://gdc.cancer.gov)
- **CAMELYON16/17**: [Grand Challenge](https://camelyon16.grand-challenge.org)
- **PANDA**: [Kaggle](https://www.kaggle.com/c/prostate-cancer-grade-assessment)


## Contact

Sanwal Ahmad Zafar — sanwal@sjtu.edu.cn
Prof. Wei Qin (Advisor) — wqin@sjtu.edu.cn
Fuxi Lab, Shanghai Jiao Tong University, China 

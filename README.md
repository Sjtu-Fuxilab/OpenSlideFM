# OpenSlideFM

<p align="center">
  <b>A Computationally Efficient Multi-Scale Foundation Model for Computational Pathology</b>
</p>

<p align="center">
  <a href="#key-results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**OpenSlideFM** is a computationally efficient foundation model for computational pathology that balances performance with practical deployment requirements. The model achieves competitive performance while requiring only **35 million parameters** and training on a **single consumer-grade GPU** (NVIDIA RTX 4090, 24 GB).

### Key Features

- 🚀 **Resource Efficient**: 35M parameters (8.8× fewer than UNI, 53× fewer than Virchow2-Giant)
- 🔬 **Multi-Scale Analysis**: Dual-resolution (0.5 μm/pixel + 2.0 μm/pixel) captures cellular morphology and tissue architecture
- ⚡ **Fast Inference**: 2.3 seconds per WSI (~26 slides/min)
- 🖥️ **Consumer Hardware**: Trainable on single RTX 4090 (24 GB VRAM)
- 📊 **Validated**: Comprehensive evaluation across 4 clinical tasks and 32,014 slides

## Key Results

### TCGA Pan-Cancer Classification

#### Full 31-Class Classification (20,000 slides, 10,795 patients)

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.21% (95% CI: 80.35-82.08%) |
| Balanced Accuracy | 76.88% ± 1.33% |
| Macro F1-Score | 75.54% ± 1.15% |
| Macro AUROC | 98.65% ± 0.09% |

#### 10-Class Benchmark Comparison (4,972 slides)

| Model | Accuracy | Balanced Acc | Macro F1 | Macro AUROC |
|-------|----------|--------------|----------|-------------|
| **OpenSlideFM** | 81.8% ± 1.9% | - | - | 0.929 ± 0.027 |
| UNI2-h | 92.04% ± 0.78% | 89.52% ± 0.92% | 88.67% ± 1.05% | 99.38% ± 0.11% |

*10-class subset: ACC, BRCA-IDC, COAD, DLBC, GBM, HNSC, KIRC, LUAD, SKCM, UCEC*

### External Validation

| Dataset | Task | Slides | Primary Metric | Result |
|---------|------|--------|----------------|--------|
| **CAMELYON16** | Lymph node metastasis detection | 270 | AUROC | 0.774 (95% CI: 0.697-0.810) |
| **CAMELYON17** | Multi-center pN staging (LOCO-CV) | 499 | Quadratic κ | 0.254 (95% CI: 0.062-0.429) |
| **PANDA** | Prostate cancer ISUP grading | 10,616 | Quadratic κ | 0.826 (95% CI: 0.810-0.842) |

#### Detailed External Validation Metrics

**CAMELYON16** (5-fold stratified CV):
- AUROC: 0.774 (95% CI: 0.697-0.810)
- Accuracy: 68.8% ± 4.2%
- F1-Score: 56.3% ± 5.8%
- Average Precision: 68.1% ± 4.1%

**CAMELYON17** (Leave-one-center-out CV):
- Quadratic κ: 0.254 (95% CI: 0.062-0.429)
- Mean CV κ: 0.204 ± 0.091
- Accuracy: 52.3% ± 8.7%
- Per-stage AUROC: pN0=0.687, pN1=0.598, pN2=0.721

**PANDA** (5-fold stratified CV, provider-stratified):
- Quadratic κ: 0.826 (95% CI: 0.810-0.842)
- Accuracy: 67.0% ± 0.9%
- Macro AUROC: 0.910 ± 0.005
- Cross-provider: Karolinska→Radboud κ=0.819, Radboud→Karolinska κ=0.823

### Ablation Studies

#### Multi-Scale Architecture (Supplementary Table 2A)

| Configuration | Accuracy | Δ vs Both |
|---------------|----------|-----------|
| **Both scales (0.5 + 2.0 μm)** | **82.32% ± 0.29%** | - |
| High-resolution only (0.5 μm) | 79.97% ± 0.20% | -2.35% (p<0.001) |
| Low-resolution only (2.0 μm) | 77.28% ± 0.56% | -5.04% (p<0.001) |

#### Token Budget (Supplementary Table 2C)

| Tokens | Accuracy |
|--------|----------|
| 100 | 62.7% |
| 500 | 65.2% |
| 1,000 | 67.8% |
| **1,600** | **82.3%** |
| 2,000 | 81.9% |

### Computational Efficiency

| Model | Parameters | Pre-training Data | GPU Memory | Hardware |
|-------|-----------|-------------------|------------|----------|
| **OpenSlideFM** | **35M** | 20K WSIs | **12 GB inf / 22 GB train** | **Consumer (RTX 4090)** |
| UNI | 307M | 100K WSIs | ~40 GB | Datacenter (A100) |
| Virchow2 | 632M | 3.1M WSIs | ~40 GB | Datacenter (A100) |
| Virchow2-Giant | 1.85B | 1.5M WSIs | ~80 GB | Datacenter (A100-80GB) |
| CONCH | 307M | 1.17M image-text | ~40 GB | Datacenter (A100) |

## Installation

```bash
# Clone repository
git clone https://github.com/Sjtu-Fuxilab/OpenSlideFM.git
cd OpenSlideFM

# Create environment
conda create -n openslidefm python=3.10
conda activate openslidefm

# Install dependencies
pip install -r requirements.txt
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools

# macOS
brew install openslide

# Windows: Download from https://openslide.org/download/
```

## Configuration

### Environment Variables

```bash
# Required
export WORKSPACE=/path/to/workspace          # Output directory
export WSI_ROOT=/path/to/wsi/slides          # WSI data directory

# Optional
export DATA_ROOT=/path/to/data               # General data directory
export PANDA_ROOT=/path/to/panda             # PANDA dataset
export RESULTS_DIR=/path/to/results          # Results output
```

## Quick Start

1. **Configure paths** via environment variables or in the notebook
2. **Open the notebook**: `jupyter notebook notebooks/OP_FM.ipynb`
3. **Run cells sequentially** - each section is documented

## Pipeline Overview

```
WSI Input → Tissue Segmentation → Two-Scale Tiling (0.5 + 2.0 μm/pixel)
         → ConvNeXt-Tiny Feature Extraction → Transformer Aggregation
         → BYOL + MFR Pre-training → Slide-level Predictions
```

### Architecture

- **Backbone**: ConvNeXt-Tiny (28M parameters)
- **Aggregator**: 4-layer Transformer encoder (hidden=768, heads=8, FFN=3072)
- **Token Budget**: 1,600 tokens (1,200 high-res + 400 low-res, 3:1 ratio)
- **Pre-training**: BYOL (EMA=0.996) + Masked Feature Reconstruction (50% mask)
- **Training**: 4 epochs, batch size 3 (grad accum 2), AdamW, LR=1.5e-4, cosine schedule

### Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `01_environment_setup.py` | Initialize workspace, validate dependencies |
| 2 | `02_dataset_manifest.py` | Create slide inventory with provenance |
| 3 | `03_quality_control.py` | Tissue detection, artifact filtering |
| 4 | `04_tiling.py` | Two-scale tile extraction (0.5 μm + 2.0 μm) |
| 5 | `05_feature_extraction.py` | ConvNeXt-Tiny 768-d features |
| 6 | `06_pretraining.py` | Self-supervised BYOL + MFR (~72 hours) |
| 7 | `08_tcga_evaluation.py` | TCGA 31-class evaluation |

## Repository Structure

```
OpenSlideFM/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── paths.yaml
├── scripts/
│   ├── 01_environment_setup.py
│   ├── 02_dataset_manifest.py
│   └── ... (21 scripts total)
└── notebooks/
    └── OP_FM.ipynb
```

## Data Availability

- **TCGA**: [NIH Genomic Data Commons](https://gdc.cancer.gov)
- **CAMELYON16/17**: [Grand Challenge](https://camelyon16.grand-challenge.org)
- **PANDA**: [Kaggle](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
- **UNI Pre-extracted Features**: [Mahmood Lab](https://huggingface.co/mahmoodlab/UNI)

## Citation

```bibtex
@article{zafar2025openslidefm,
  title={OpenSlideFM: A Computationally Efficient Multi-Scale Foundation 
         Model for Computational Pathology},
  author={Zafar, Sanwal Ahmad and Qin, Wei},
  journal={arXiv preprint},
  year={2025},
  institution={Shanghai Jiao Tong University}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE)

## Contact

- **Sanwal Ahmad Zafar** - sanwal@sjtu.edu.cn
- **Wei Qin** (Advisor) - Shanghai Jiao Tong University, Department of Industrial Engineering and Management

---

<p align="center">Made with ❤️ at SJTU Fuxi Lab</p>

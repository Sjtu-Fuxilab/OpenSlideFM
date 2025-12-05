# OpenSlideFM

<p align="center">
  <b>A Resource-Efficient Foundation Model for Computational Pathology on Whole Slide Images</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#pipeline">Pipeline</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**OpenSlideFM** is a foundation model designed for computational pathology that achieves competitive performance with significantly fewer parameters than existing models, enabling deployment on consumer-grade hardware (single GPU training).

### Key Features

- 🚀 **Resource Efficient**: Trainable on a single RTX 4090 GPU
- 🎯 **Competitive Performance**: Matches larger foundation models on standard benchmarks
- 🔬 **Multi-scale Analysis**: Two-scale tiling strategy for comprehensive tissue analysis
- 📊 **Reproducible**: Complete pipeline with provenance tracking

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- OpenSlide

### Setup

```bash
# Clone repository
git clone https://github.com/Sjtu-Fuxilab/OpenSlideFM.git
cd OpenSlideFM

# Create conda environment
conda create -n openslidefm python=3.10
conda activate openslidefm

# Install dependencies
pip install -r requirements.txt

# Install OpenSlide (system dependency)
# Ubuntu/Debian:
sudo apt-get install openslide-tools
# macOS:
brew install openslide
# Windows: Download from https://openslide.org/download/
```

## Quick Start

### 1. Configure Paths

Edit `configs/paths.yaml` to set your data directories:

```yaml
wsi_root: /path/to/your/slides
workspace: /path/to/output
```

### 2. Run Pipeline

```bash
# Step 1: Environment setup & validation
python scripts/01_environment_setup.py

# Step 2: Create dataset manifest
python scripts/02_dataset_manifest.py

# Step 3: Quality control
python scripts/03_quality_control.py

# Step 4: Tile extraction
python scripts/04_tiling.py

# Step 5: Feature extraction
python scripts/05_feature_extraction.py

# Step 6: Pretraining
python scripts/06_pretraining.py
```

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenSlideFM Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  WSI Input    →  QC & Mask  →  Two-Scale   →  Feature    →     │
│  (SVS/TIFF)      Generation     Tiling        Extraction        │
│                                                                 │
│              →  BYOL         →  Slide       →  Downstream       │
│                 Pretraining     Embeddings     Tasks            │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Scripts

| Script | Description |
|--------|-------------|
| `01_environment_setup.py` | Initialize workspace, validate dependencies |
| `02_dataset_manifest.py` | Create slide inventory with provenance |
| `03_quality_control.py` | Tissue detection, artifact filtering |
| `04_tiling.py` | Two-scale tile extraction (5x, 20x) |
| `05_feature_extraction.py` | Extract tile-level features |
| `06_pretraining.py` | BYOL-based self-supervised learning |
| `07_checkpoint_save.py` | Export trained encoder |
| `08_tcga_evaluation.py` | TCGA cancer type classification |

## Benchmarks

### TCGA Pan-Cancer Classification (31 classes)

| Model | Parameters | Accuracy | Hardware |
|-------|-----------|----------|----------|
| UNI | 307M | 81.2% | 8× A100 |
| CONCH | 307M | 79.8% | 8× A100 |
| **OpenSlideFM** | **42M** | **80.1%** | **1× RTX 4090** |

### CAMELYON16 (Tumor Detection)

| Model | AUC | Accuracy |
|-------|-----|----------|
| UNI | 0.942 | 89.3% |
| **OpenSlideFM** | **0.938** | **88.7%** |

## Project Structure

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
│   ├── ...
│   └── 21_results_extraction.py
└── notebooks/
    └── OP_FM.ipynb
```

## Data

This model was trained on:
- **TCGA**: 20,000 diagnostic slides across 31 cancer types
- **CAMELYON16/17**: Lymph node metastasis detection
- **PANDA**: Prostate cancer grading

## Citation

If you use OpenSlideFM in your research, please cite:

```bibtex
@article{zafar2025openslidefm,
  title={OpenSlideFM: A Resource-Efficient Foundation Model for 
         Computational Pathology on Whole Slide Images},
  author={Zafar, Sanwal Ahmad and Qin, Wei},
  journal={arXiv preprint},
  year={2025},
  institution={Shanghai Jiao Tong University}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Shanghai Jiao Tong University, Fuxi Lab
- TCGA Research Network
- OpenSlide project

## Contact

- **Sanwal Ahmad Zafar** - sanwal@sjtu.edu.cn
- **Wei Qin** (Advisor) - Shanghai Jiao Tong University

---

<p align="center">
  Made with ❤️ at SJTU Fuxi Lab
</p>

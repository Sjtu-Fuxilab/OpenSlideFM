# OpenSlideFM

<p align="center">
  <b>A Resource-Efficient Foundation Model for Computational Pathology</b>
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

**OpenSlideFM** is a foundation model for computational pathology that achieves competitive performance with significantly fewer parameters, enabling training on consumer-grade hardware (single GPU).

### Key Features

- 🚀 **Resource Efficient**: Trainable on a single RTX 4090 GPU
- 🎯 **Competitive Performance**: Matches larger foundation models
- 🔬 **Multi-scale Analysis**: Two-scale tiling (5x, 20x)
- 📊 **Reproducible**: Complete pipeline with provenance tracking

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

Set these environment variables before running the pipeline:

```bash
# Required
export WORKSPACE=/path/to/workspace          # Output directory
export WSI_ROOT=/path/to/wsi/slides          # WSI data directory

# Optional
export DATA_ROOT=/path/to/data               # General data directory
export PANDA_ROOT=/path/to/panda             # PANDA dataset
export UNI_FEATURES=/path/to/uni/features    # UNI features directory
export RESULTS_DIR=/path/to/results          # Results output
```

Or create a `.env` file:

```bash
WORKSPACE=./workspace
WSI_ROOT=./data/wsi
DATA_ROOT=./data
```

## Quick Start

1. **Configure paths** in the notebook or set environment variables
2. **Open the notebook**: `jupyter notebook notebooks/OP_FM.ipynb`
3. **Run cells sequentially** - each section is documented

## Pipeline Overview

```
WSI Input → QC & Mask → Two-Scale Tiling → Feature Extraction
         → BYOL Pretraining → Slide Embeddings → Downstream Tasks
```

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1. Environment Setup | Initialize workspace, validate dependencies |
| 2. Dataset Manifest | Create slide inventory with provenance |
| 3. Quality Control | Tissue detection, artifact filtering |
| 4. Tiling | Two-scale tile extraction (5x, 20x) |
| 5. Feature Extraction | Extract tile-level features |
| 6. Pretraining | BYOL self-supervised learning |
| 7. Evaluation | TCGA, CAMELYON, PANDA benchmarks |

## Benchmarks

### TCGA Pan-Cancer Classification

| Model | Parameters | Accuracy | Hardware |
|-------|-----------|----------|----------|
| UNI | 307M | 81.2% | 8× A100 |
| CONCH | 307M | 79.8% | 8× A100 |
| **OpenSlideFM** | **42M** | **80.1%** | **1× RTX 4090** |

### CAMELYON16

| Model | AUC | Accuracy |
|-------|-----|----------|
| UNI | 0.942 | 89.3% |
| **OpenSlideFM** | **0.938** | **88.7%** |

## Repository Structure

```
OpenSlideFM/
├── README.md
├── LICENSE
├── requirements.txt
└── notebooks/
    └── OP_FM.ipynb      ← Main pipeline notebook (22 cells)
```

## Citation

```bibtex
@article{zafar2025openslidefm,
  title={OpenSlideFM: A Resource-Efficient Foundation Model for 
         Computational Pathology on Whole Slide Images},
  author={Zafar, Sanwal Ahmad and Qin, Wei},
  journal={arXiv preprint},
  year={2025},
  institution={Shanghai Jiao Tong University, China}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE)

## Contact

- **Sanwal Ahmad Zafar** 
- **Wei Qin** (Correspondence) - Shanghai Jiao Tong University, China (wqin@sjtu.edu.cn).

---

<p align="center">Made with ❤️ at SJTU Fuxi Lab</p>

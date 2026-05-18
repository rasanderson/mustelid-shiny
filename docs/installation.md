---
description: "How to install and set up MegaDetector-Classifier for camera-trap species classification fine-tuning."
tags:
  - MegaDetector-Classifier installation
  - camera-trap classification setup
  - PyTorch-Wildlife
  - fine-tuning setup
  - wildlife monitoring
---

# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, but recommended for faster training)

## Install with pip

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
pip install -r requirements.txt
```

This installs the following dependencies:

| Package | Purpose |
|---|---|
| `PytorchWildlife` | Core models and framework integration |
| `lightning` | Training loop and checkpointing (PyTorch Lightning) |
| `scikit_learn` | Data splitting utilities |
| `munch` | YAML config as dot-accessible object |
| `typer` | CLI argument handling |

## Install with conda

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
conda env create -f environment.yaml
conda activate PT_Finetuning
```

## Verify

```python
from PytorchWildlife.models import classification as pw_classification
print("MegaDetector-Classifier is ready.")
```

## GPU Setup

Training on GPU is recommended for faster iteration. Verify CUDA availability:

```python
import torch
print(torch.cuda.is_available())  # should print True on a CUDA-enabled machine
```

CPU training is supported but will be slower for larger datasets.

## Next Steps

- Download [demo data](https://zenodo.org/records/15376499/files/demo_data_clf.zip) for an immediate end-to-end test
- Read the [Usage Guide](usage.md) for a full walkthrough of data preparation, configuration, and training

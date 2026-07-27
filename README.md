# MegaDetector-Classifier

Microsoft AI for Good Lab's open-source classification fine-tuning tool — train custom species classifiers on your own camera-trap datasets and deploy them through PyTorch-Wildlife.

[![License](https://img.shields.io/github/license/microsoft/MegaDetector-Classifier)](https://github.com/microsoft/MegaDetector-Classifier/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch-Wildlife](https://img.shields.io/badge/PyTorch--Wildlife-ecosystem-green.svg)](https://github.com/microsoft/Biodiversity)

MegaDetector-Classifier is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) ecosystem and is powered by the [PyTorch-Wildlife](https://github.com/microsoft/PytorchWildlife) framework. It is free, open-source, and available under the MIT license.

## Part of the Biodiversity Ecosystem

MegaDetector-Classifier is one tool in a larger open-source ecosystem from the Microsoft AI for Good Lab. Each project lives in its own repository, with the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) umbrella tying them together.

| Repository | Description |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | The umbrella repository — documentation hub for the AI for Good Lab's biodiversity work |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal, human, and vehicle detection for camera-trap images |
| [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife) | The collaborative deep learning framework that hosts MegaDetector, species classifiers, and demo notebooks |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic AI for audio-based wildlife detection and classification |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | Wildlife detection in aerial and drone imagery |
| [microsoft/MegaDetector-Sonar](https://github.com/microsoft/MegaDetector-Sonar) | Sonar-based wildlife detection for aquatic monitoring |
| [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | **This repo** — classification fine-tuning for camera-trap species identification |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI-enabled edge device for field recording |

## Overview

MegaDetector-Classifier is a training toolkit for fine-tuning ResNet-based species classifiers on custom camera-trap image datasets. The output weights integrate directly with the [PyTorch-Wildlife](https://github.com/microsoft/PytorchWildlife) framework, making it straightforward to deploy a classifier trained on your own data.

**Key capabilities:**
- ResNet-18 and ResNet-50 classifier training using PyTorch Lightning
- Three data-splitting strategies designed for camera-trap realities: random, location-based, and sequence-based
- YAML-based configuration — no code changes required for most use cases
- Demo data included for immediate testing without your own dataset

### How ResNet Fine-Tuning Works In This Repo

MegaDetector-Classifier trains a single ResNet-based classifier per run, selected by `num_layers` in `configs/config.yaml`:

- `num_layers: 18` → ResNet-18 backbone
- `num_layers: 50` → ResNet-50 backbone

The training model (`PlainResNetClassifier`) uses transfer learning by initializing the feature extractor from ImageNet pretrained torchvision ResNet weights, then attaching a new classification head sized to your `num_classes`.

Important behavior details:

- Only one backbone is used per run (ResNet-18 **or** ResNet-50, not both simultaneously)
- Both the backbone (feature extractor) and classifier head are optimized during training
- Backbone and head use separate optimizer parameter groups (`lr_feature` and `lr_classifier` settings)

**Designed for:**
- Conservation practitioners adapting existing classifiers to new geographic regions
- Researchers adding new species to the PyTorch-Wildlife model zoo
- Projects running MegaDetector detection upstream and needing a matched classifier downstream

## Installation

### Using pip

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
pip install -r requirements.txt
```

### Using conda

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
conda env create -f environment.yaml
conda activate PT_Finetuning
```

**Requirements:** Python 3.9+

## Quick Start

1. Configure `configs/config.yaml` — set `dataset_root`, `annotation_dir`, `num_classes`, and `split_type`
2. Choose the transfer-learning backbone with `num_layers`:

```yaml
# one model per run: choose exactly one
num_layers: 18   # ResNet-18
# num_layers: 50 # ResNet-50
weights_init: ImageNet
```

3. Run training:

```bash
python main.py
```

To train on images that are already pre-cropped, set:

```yaml
dataset_name: Custom_PreCropped
enable_auto_cropping: False
```

Output weights are saved to the `weights/` directory and can be loaded directly into PyTorch-Wildlife.

## Data Preparation

### Data Structure

Images should be stored in a single flat directory (no nested subdirectories). An `annotations.csv` file — placed outside the images directory — maps each image to its class:

```plaintext
MegaDetector-Classifier/
├── data/
│   ├── imgs/                       # All images stored here (flat)
│   └── annotation_example.csv      # Annotations file
└── configs/config.yaml
```

### Annotation File Format

The CSV must contain three columns:

| Column | Description | Example |
|---|---|---|
| `path` | Relative path to the image | `imgs/leopard_001.jpg` |
| `classification` | Integer class ID | `0` |
| `label` | Human-readable class name | `leopard` |

### Data Splitting

MegaDetector-Classifier supports three splitting strategies, selected via `split_type` in `config.yaml`:

| Strategy | When to use | Extra column required |
|---|---|---|
| `random` | Balanced class distribution; not recommended for camera-trap bursts | None |
| `location` | Keeps all images from one camera location in the same split | `Location` |
| `sequence` | Groups burst images within 30-second windows before splitting | `Photo_time` (YYYY-MM-DD HH:MM:SS) |

> **Camera-trap note:** Random splitting is not recommended because burst images of the same animal can appear in both training and validation sets, causing artificially high validation accuracy. Use `location` or `sequence` splitting instead.

### Demo Data

Download demo data to test the pipeline without your own dataset:

```bash
# Download and extract
wget https://zenodo.org/records/15376499/files/demo_data_clf.zip
unzip demo_data_clf.zip -d data/
```

Then set `dataset_root: ./data/imgs` in `configs/config.yaml` and run `python main.py`.

## Loading PytorchWildlife classifiers

In addition to training a custom classifier, MegaDetector-Classifier ships a PyTorch-Wildlife–compatible inference layer. Pretrained species classifiers can be loaded with a single import — the same `classification` subpackage layout that PyTorch-Wildlife exposes:

```python
from src.models import classification as pw_classification

# Pick any of the available loaders below
model = pw_classification.AI4GAmazonRainforest(device="cuda")
results = model.single_image_classification("path/to/image.jpg")
```

Available loaders:

- `AI4GAmazonRainforest` — Amazon Rainforest species classifier (ResNet-50, v1/v2 weights)
- `AI4GOpossum` — binary opossum / not-opossum classifier (ResNet-50)
- `AI4GSnapshotSerengeti` — Snapshot Serengeti 10-class classifier (ResNet-18)
- `CustomWeights` — load your own ResNet weights (e.g. those produced by `main.py`)
- `DeepfauneClassifier` — Deepfaune ViT-L/14 DINOv2 classifier (timm backbone)
- `DFNE` — DFNE ViT-L/14 DINOv2 classifier (timm backbone)
- `SpeciesNetTFInference` — SpeciesNet TensorFlow classifier (optional, see below)

### Optional: SpeciesNet

`SpeciesNetTFInference` depends on the `speciesnet` package, which pulls in TensorFlow. It is **not** installed by `requirements.txt` / `environment.yaml`. To enable it:

```bash
pip install speciesnet
```

If `speciesnet` is not installed, `from src.models import classification` still succeeds — only instantiating `SpeciesNetTFInference()` raises an `ImportError` with installation instructions.

## Repository Structure

```
MegaDetector-Classifier/
├── main.py                          # Training entry point
├── requirements.txt                 # pip dependencies
├── environment.yaml                 # conda environment
├── configs/
│   └── config.yaml                  # Training configuration
└── src/
    ├── algorithms/
    │   └── plain.py                 # Training algorithm (PyTorch Lightning)
    ├── data/                        # Inference data layer (PyTorch-Wildlife-compatible)
    │   ├── transforms.py            # Classification / MegaDetector transforms
    │   └── datasets.py              # ImageFolder / DetectionCrops loaders
    ├── datasets/
    │   └── custom.py                # Custom training dataset loader
    ├── models/
    │   ├── plain_resnet.py          # ResNet-18/50 training classifier
    │   └── classification/          # Inference classifiers (PyTorch-Wildlife-compatible)
    │       ├── base_classifier.py
    │       ├── resnet_base/         # AI4GAmazon, AI4GOpossum, Serengeti, CustomWeights
    │       ├── timm_base/           # DeepfauneClassifier, DFNE
    │       └── speciesnet_base/     # SpeciesNetTFInference (optional)
    └── utils/
        ├── batch_detection_cropping.py   # MegaDetector crop integration
        ├── data_splitting.py             # Random / location / sequence splits
        └── utils.py                      # Shared utilities
```

## Citation

If you use MegaDetector-Classifier in your research, please cite the PyTorch-Wildlife paper:

```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

You can also cite this software directly using the [`citation.cff`](citation.cff) file in this repository.

## Contributing

Issues, feature requests, and pull requests are welcome at [microsoft/MegaDetector-Classifier/issues](https://github.com/microsoft/MegaDetector-Classifier/issues).

For framework-level changes (PyTorch-Wildlife API, models, datasets), see [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife). For ecosystem-wide questions, see the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) umbrella.

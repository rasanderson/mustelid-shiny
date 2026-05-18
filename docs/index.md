---
description: "MegaDetector-Classifier — Microsoft AI for Good Lab's open-source classification fine-tuning tool for training custom species classifiers on camera-trap datasets."
tags:
  - MegaDetector-Classifier
  - camera-trap classification
  - species identification
  - fine-tuning
  - transfer learning
  - PyTorch-Wildlife
  - conservation AI
---

# Microsoft MegaDetector-Classifier

**Open-source classification fine-tuning for camera-trap species identification.**

MegaDetector-Classifier is a training toolkit from the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/) for fine-tuning ResNet-based species classifiers on custom camera-trap image datasets. Output weights integrate directly with the [PyTorch-Wildlife](https://github.com/microsoft/PytorchWildlife) framework. It is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) ecosystem.

---

## What It Does

MegaDetector-Classifier takes your labeled camera-trap images and produces a trained classifier ready to deploy in PyTorch-Wildlife:

1. **Data preparation** — flat image directory + a simple `annotations.csv`; three splitting strategies (random, location, sequence) designed for camera-trap realities
2. **Training** — ResNet-18 or ResNet-50 classifiers via PyTorch Lightning, configured entirely through `config.yaml`
3. **Output** — trained weights saved to `weights/`, ready to load into PyTorch-Wildlife for inference

---

## Get Started

See the [Installation](installation.md) page, then configure and run:

```bash
git clone https://github.com/microsoft/MegaDetector-Classifier
cd MegaDetector-Classifier
pip install -r requirements.txt
# edit configs/config.yaml, then:
python main.py
```

Demo data is available on [Zenodo](https://zenodo.org/records/15376499/files/demo_data_clf.zip) for immediate testing without your own dataset. See the [Usage Guide](usage.md) for a full walkthrough.

---

## Part of the Biodiversity Ecosystem

MegaDetector-Classifier is one tool in a larger open-source ecosystem from the Microsoft AI for Good Lab.

| Repository | Description |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | Umbrella hub — PyTorch-Wildlife, MegaDetector, ecosystem overview |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal, human, and vehicle detection for camera-trap images |
| [microsoft/PytorchWildlife](https://github.com/microsoft/PytorchWildlife) | The collaborative deep learning framework for wildlife monitoring |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic AI for audio-based wildlife detection and classification |
| [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | **This repo** — classification fine-tuning for camera-trap species identification |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI-enabled edge device |

---
description: "How to use MegaDetector-Classifier — data preparation, splitting strategies, configuration, training, and integrating output weights with PyTorch-Wildlife."
tags:
  - MegaDetector-Classifier usage
  - camera-trap classification training
  - data splitting
  - config.yaml
  - species classification fine-tuning
  - PyTorch-Wildlife integration
---

# Usage Guide

This page walks through the full MegaDetector-Classifier workflow: preparing your data, configuring training, running the classifier, and using the output weights.

---

## 1. Data Structure

Images must be stored in a single **flat directory** (no subdirectories). An `annotations.csv` file placed alongside (not inside) the images directory maps each image to its class.

```plaintext
MegaDetector-Classifier/
├── data/
│   ├── imgs/                       # All images stored here (flat — no nested folders)
│   └── annotation_example.csv      # Annotations file
└── configs/config.yaml
```

### Annotation File Format

The CSV must contain exactly these three columns:

| Column | Type | Description | Example |
|---|---|---|---|
| `path` | string | Relative path to the image from the CSV location | `imgs/leopard_001.jpg` |
| `classification` | integer | Unique integer ID for each class | `0` |
| `label` | string | Human-readable class name | `leopard` |

---

## 2. Data Splitting

Set `split_data: True` in `config.yaml` to have MegaDetector-Classifier split your annotations into train, validation, and test sets automatically. The splitting strategy is controlled by `split_type`.

### Splitting Strategies

| Strategy | When to use | Extra column required |
|---|---|---|
| `random` | General-purpose balanced split | None |
| `location` | Keeps all images from one camera location in the same split | `Location` |
| `sequence` | Groups burst images within 30-second windows before splitting | `Photo_time` (YYYY-MM-DD HH:MM:SS) |

> **Important — camera-trap burst images:** Camera traps frequently capture bursts of images of the same animal within seconds. With random splitting, nearly identical frames can end up in both training and validation sets, causing artificially inflated validation accuracy (overfitting). Use `location` or `sequence` splitting to prevent this.

If you already have pre-split CSV files for train/val/test, set `split_data: False` and point `annotation_dir` to the directory containing those files.

---

## 3. Configuration

All training parameters live in `configs/config.yaml`. Edit this file before running `python main.py`.

### Training Parameters

| Parameter | Description | Default |
|---|---|---|
| `conf_id` | Unique identifier for this training run | `Crop_Res18_plain_071824` |
| `algorithm` | Training algorithm | `Plain` |
| `log_dir` | Directory for training logs | `Crop` |
| `num_epochs` | Total training epochs | `30` |
| `log_interval` | How often to log training info (in steps) | `10` |
| `parallel` | Set to `1` to enable multi-GPU training | `0` |

### Data Parameters

| Parameter | Description |
|---|---|
| `dataset_root` | Root directory where images are stored |
| `dataset_name` | Dataset type (`Custom_Crop` or `Custom_PreCropped`) |
| `enable_auto_cropping` | `True` runs MegaDetector cropping before training; set `False` for pre-cropped inputs |
| `cropped_images_dir` | Output/input directory used by `Custom_Crop` (default: `cropped_resized`) |
| `annotation_dir` | Directory containing annotation CSV files |
| `split_path` | Path to single CSV for auto-splitting |
| `test_size` | Proportion of data for test set (e.g. `0.2`) |
| `val_size` | Proportion of data for validation set (e.g. `0.2`) |
| `split_data` | `True` to auto-split, `False` if splits already exist |
| `split_type` | `random`, `location`, or `sequence` |
| `batch_size` | Images per batch (default: `32`) |
| `num_workers` | Dataloader worker processes (default: `4`) |

### Model Parameters

| Parameter | Description |
|---|---|
| `num_classes` | Number of species classes in your dataset |
| `model_name` | Architecture (`PlainResNetClassifier`) |
| `num_layers` | ResNet depth — `18` or `50` |
| `weights_init` | Initial weights — `ImageNet` for transfer learning |

### ResNet Implementation Details (Explicit)

Training uses `PlainResNetClassifier`, which builds a torchvision-style ResNet backbone plus a dataset-specific linear classifier head.

- Set `num_layers: 18` to use a ResNet-18 backbone
- Set `num_layers: 50` to use a ResNet-50 backbone
- Exactly one backbone is active per run (not both at once)

Transfer-learning behavior:

- The feature extractor is initialized from ImageNet pretrained ResNet weights
- A new classifier head is created for your configured `num_classes`
- Both feature extractor and classifier head are fine-tuned during training
- Feature extractor and classifier head use separate optimization parameter groups (`lr_feature` vs `lr_classifier` and related momentum/weight decay settings)

Example backbone selection in `configs/config.yaml`:

```yaml
# choose one per run
num_layers: 18   # ResNet-18
# num_layers: 50 # ResNet-50
weights_init: ImageNet
```

### Optimization Parameters

| Parameter | Description |
|---|---|
| `lr_feature` | Learning rate for the feature extractor |
| `momentum_feature` | Momentum for the feature extractor optimizer |
| `weight_decay_feature` | Weight decay for the feature extractor |
| `lr_classifier` | Learning rate for the classifier head |
| `momentum_classifier` | Momentum for the classifier head optimizer |
| `weight_decay_classifier` | Weight decay for the classifier head |
| `step_size` | LR scheduler step size (epochs) |
| `gamma` | LR scheduler decay factor |

> **Architecture note:** The current version supports only `PlainResNetClassifier` with ResNet-18 or ResNet-50 backbones. The classifier head and feature extractor are trained with separate optimizers, which is required for compatibility with the PyTorch-Wildlife framework.

### Cropping Modes

You can run training in either crop mode or pre-cropped mode:

```yaml
# Option 1: automatic MegaDetector cropping (default)
dataset_name: Custom_Crop
enable_auto_cropping: True
cropped_images_dir: cropped_resized
```

```yaml
# Option 2: images are already pre-cropped (no additional cropping)
dataset_name: Custom_PreCropped
enable_auto_cropping: False
```

In pre-cropped mode, training reads `train_annotations.csv` and `val_annotations.csv` (automatically generated if e.g. random split is used) from `dataset_root` and does not call the crop utility.

---

## 4. Running Training

After configuring `configs/config.yaml`:

```bash
python main.py
```

Monitor the console output for per-epoch loss and accuracy. Logs are written to the directory specified in `log_dir`.

### Quick test with demo data

```bash
# Download demo data
wget https://zenodo.org/records/15376499/files/demo_data_clf.zip
unzip demo_data_clf.zip -d data/

# The demo config already points to ./data/imgs — just run:
python main.py
```

---

## 5. Output

Trained weights are saved to the `weights/` directory at the end of training. These weights follow the PyTorch-Wildlife classifier interface and can be loaded directly into the framework for inference on new images.

The run directory under `log/` also receives these CSV files next to `loss_accuracy.csv`:

- `train_predictions.csv`
- `test_predictions.csv`
- `train_confusion_matrix.csv`
- `test_confusion_matrix.csv`

The prediction CSVs preserve the annotation metadata and add six probability columns (`prob_class_0` through `prob_class_5`). The confusion-matrix CSVs use the same six-class order on both axes so train and test results are directly comparable.

---

## 6. Integration with MegaDetector

A common workflow pairs MegaDetector detection upstream with MegaDetector-Classifier downstream:

1. Run [MegaDetector](https://github.com/microsoft/MegaDetector) on your camera-trap images to detect and crop animals
2. Use `src/utils/batch_detection_cropping.py` to generate cropped images from MegaDetector outputs
3. Train MegaDetector-Classifier on the cropped detections
4. Deploy the resulting classifier through [PyTorch-Wildlife](https://github.com/microsoft/PytorchWildlife)

If you already have pre-cropped inputs, use `Custom_PreCropped` and set `enable_auto_cropping: False` to skip this step.

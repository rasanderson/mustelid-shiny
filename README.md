# mustelid-shiny

Inference-only pipeline for identifying UK mustelid species (mink, otter, pine
marten, polecat, stoat, weasel) in camera-trap images.

The pipeline has two stages:

1. **`deepfaune/extractMustelidCrop.py`** — runs the [DeepFaune](https://www.deepfaune.cnrs.fr/)
   detector/classifier on a single image. If DeepFaune predicts `mustelid`,
   the animal is cropped out and saved to an output directory, along with a
   CSV manifest describing the detection.
2. **`predict_image.py`** — classifies a single (cropped) image with a
   fine-tuned ResNet-50 mustelid classifier, producing a species prediction
   and per-class confidences as JSON.

## Installation

```bash
conda env create -n mustelid-shiny --file requirements.inference.lock.txt
conda activate mustelid-shiny
```

## Usage

```bash
# 1. Detect and crop a mustelid from a raw camera-trap image
python deepfaune/extractMustelidCrop.py path/to/image.jpg output_dir/

# 2. Classify the cropped image
python predict_image.py output_dir/image_mustelid.jpg
```

`predict_image.py` accepts `--device {cpu,cuda}`, `--checkpoint` (defaults to
the ResNet-50 checkpoint under `weights/`), `--img-id`, and `--output` to
save the JSON result to a file instead of printing it.

## Repository Structure

```
mustelid-shiny/
├── deepfaune/                # DeepFaune detector/classifier (inference only)
├── predict_image.py           # Mustelid species classifier entry point
├── weights/                   # Fine-tuned classifier checkpoint
├── requirements.inference.lock.txt   # Pinned inference dependencies
└── src/
    ├── data/                  # PyTorch-Wildlife-compatible transforms/datasets
    └── models/
        └── classification/
            └── resnet_base/   # CustomWeights ResNet classifier used at inference
```

## Citation

See [`citation.cff`](citation.cff).

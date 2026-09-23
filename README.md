# mustelid-shiny

Inference-only pipeline for identifying UK mustelid species (mink, otter, pine
marten, polecat, stoat, weasel) in camera-trap images.

## Two-stage pipeline (debugging)
The pipeline has two stages:

1. **`deepfaune/extractMustelidCrop.py`** — runs the [DeepFaune](https://www.deepfaune.cnrs.fr/)
   detector/classifier on a single image. If DeepFaune predicts `mustelid`,
   the animal is cropped out and saved to an output directory, along with a
   CSV manifest describing the detection.
2. **`predict_image.py`** — classifies a single (cropped) image with a
   fine-tuned ResNet-50 mustelid classifier, producing a species prediction
   and per-class confidences as JSON.

## Single-stage pipeline (deployment)
**`predict_mustelid.py`** uses the DeepFaune model and if it detects a mustelid the bounding boxes are passed through to the 6-species classifier. A CSV is generated as output, with bounding box of detected mustelid, plust the 6 probability values. If DeepFaune does not detect a mustelid then all probabilities are tagged as 0.

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

# Running locally
It does not appear to be posssible (despite documentation) to run Shiny for Python apps from VS Code simply by clicking on the "Run" button. Instead, in a Terminal, need:

```
mamba activate mustelid-shiny
shiny run app.py
```

This gives a local URL which can be opened both within VS Code and an external browser to test the code locally.

# Deployment to shinyapps.io
**Note:** Posit have announced that they are migrating hosted apps from shinyapps.io to Posit Connect Cloud by end of December 2026 so these instructions may change slightly. The application is very large, due to the model weights files, at about 700 Mb, but this is well-within the shinyapps.io 5 Gb limit. However, model inference requires that available memory on the shinyapps.io application be set at maximum after it has been deployed. As the app is big, the TIMEOUT option needs to be exported else deployment times out mid-way through the process:

```
mamba activate mustelid-shiny
export CONNECT_REQUEST_TIMEOUT=36000
rsconnect deploy shiny ~/pete/mustelid/mustelid-shiny --name naturalandenvironmentalscience --title mustelid
```

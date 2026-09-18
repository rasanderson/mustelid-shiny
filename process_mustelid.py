"""Detect a mustelid in an image and classify its species in one pass.

Combines the DeepFaune detector (deepfaune/extractMustelidCrop.py) with the
fine-tuned 6-species classifier (predict_image.py), passing the detected crop
directly in memory instead of round-tripping through a saved JPEG.

Usage:
    python predict_mustelid.py IMAGE_PATH OUTPUT_DIRECTORY
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent / "deepfaune"))

from predictTools import PredictorImage, txt_empty, txt_undefined  # noqa: E402
from detectTools import cropSquareCVtoPIL  # noqa: E402

from src.models import classification as pw_classification


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULT_CHECKPOINT = (
    "weights/Crop/Plain/"
    "Crop_Res50_plain_071824-0-epoch=14-valid_mac_acc=81.92.ckpt"
)
CLASS_NAMES = [
    "mink",
    "otter",
    "pinemarten",
    "polecat",
    "stoat",
    "weasel",
]

# Caches so repeated calls (e.g. from a running web app) don't reload weights from disk
_deepfaune_models_cache = {}
_species_model_cache = {}


def get_deepfaune_models(detector_name, device, birdclassification=True):
    """Build (and cache) the DeepFaune detector + classifier for a given device."""
    from predictTools import Classifier, ClassifierWithBirds
    from detectTools import Detector

    key = (detector_name, device, birdclassification)
    if key not in _deepfaune_models_cache:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device in (None, "auto") else torch.device(device)
        detector = Detector(name=detector_name, device=torch_device)
        classifier = ClassifierWithBirds(torch_device) if birdclassification else Classifier(torch_device)
        _deepfaune_models_cache[key] = (detector, classifier)
    return _deepfaune_models_cache[key]


def get_species_model(checkpoint, device):
    """Build (and cache) the mustelid species classifier for a given device."""
    key = (checkpoint, device)
    if key not in _species_model_cache:
        _species_model_cache[key] = pw_classification.CustomWeights(
            weights=checkpoint, class_names=CLASS_NAMES, device=device,
        )
    return _species_model_cache[key]


def classify_result_category(deepfaune_prediction):
    """Map a DeepFaune prediction to 'no_animal', 'mustelid', or 'other'."""
    if deepfaune_prediction == txt_empty["en"] or deepfaune_prediction == txt_undefined["en"]:
        return "no_animal"
    if deepfaune_prediction == "mustelid":
        return "mustelid"
    return "other"


def classify_mustelid_image(
    image_path,
    checkpoint=DEFAULT_CHECKPOINT,
    threshold=0.5,
    maxlag=20,
    detector="DFbsMDS",
    device=None,
    species_device="cpu",
):
    """Run DeepFaune detection and, if a mustelid is found, species classification.

    Returns the same row dict schema written to the CSV manifest by `main()`.
    """
    image_path = str(image_path)
    deepfaune_detector, deepfaune_classifier = get_deepfaune_models(detector, device)
    species_model = get_species_model(checkpoint, species_device)

    predictor = PredictorImage(
        filenames=[image_path],
        threshold=threshold,
        maxlag=maxlag,
        LANG="en",
        birdclassification=True,
        detectorname=detector,
        device=device,
        detector=deepfaune_detector,
        classifier=deepfaune_classifier,
    )
    predictor.allBatch()

    filename = predictor.getFilenames()[0]
    predicted_classes, scores, boxes, counts = predictor.getPredictions()
    predicted_class = predicted_classes[0]
    score = scores[0]
    box = boxes[0]
    count = counts[0]

    row = {
        "source_image": filename,
        "crop_image": "",
        "deepfaune_prediction": predicted_class,
        "deepfaune_score": float(score),
        "animal_count": int(count),
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "species_prediction": "",
        "species_confidence": "",
    }
    for class_name in CLASS_NAMES:
        row[f"prob_{class_name}"] = 0.0

    if predicted_class == "mustelid":
        image = cv2.imread(filename)
        if image is None:
            raise RuntimeError(f"Could not read image: {filename}")

        crop = cropSquareCVtoPIL(image, box)
        result = species_model.single_image_classification(
            np.array(crop), img_id=Path(image_path).name
        )

        row["species_prediction"] = result["prediction"]
        row["species_confidence"] = result["confidence"]
        for class_name, confidence in result["all_confidences"]:
            row[f"prob_{class_name}"] = confidence
        row["_crop"] = crop

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Detect a mustelid and classify its species in a single pass."
    )
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for the CSV manifest (and optional crop)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--maxlag", type=float, default=20)
    parser.add_argument("--detector", default="DFbsMDS")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the species classifier checkpoint",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Also save the mustelid crop to output_dir (not required for classification)",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.image.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error(f"Unsupported image extension: {args.image.suffix}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CustomWeights requires an explicit device string, unlike PredictorImage's "auto".
    species_device = args.device or "cpu"

    row = classify_mustelid_image(
        args.image,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        maxlag=args.maxlag,
        detector=args.detector,
        device=args.device,
        species_device=species_device,
    )
    crop = row.pop("_crop", None)
    predicted_class = row["deepfaune_prediction"]

    if predicted_class == "mustelid":
        if args.save_crop:
            crop_path = args.output_dir / f"{args.image.stem}_mustelid.jpg"
            crop.save(crop_path, format="JPEG")
            row["crop_image"] = str(crop_path)
            print(f"Saved mustelid crop to {crop_path}")
    else:
        print(f"No species classification run; DeepFaune predicted: {predicted_class}")

    manifest_path = args.output_dir / "mustelid_prediction.csv"
    pd.DataFrame([row]).to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()


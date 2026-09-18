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

sys.path.append(str(Path(__file__).resolve().parent / "deepfaune"))

from predictTools import PredictorImage  # noqa: E402
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

    predictor = PredictorImage(
        filenames=[str(args.image)],
        threshold=args.threshold,
        maxlag=args.maxlag,
        LANG="en",
        birdclassification=True,
        detectorname=args.detector,
        device=args.device,
    )
    species_model = pw_classification.CustomWeights(
        weights=args.checkpoint,
        class_names=CLASS_NAMES,
        device=species_device,
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
            np.array(crop), img_id=args.image.name
        )

        row["species_prediction"] = result["prediction"]
        row["species_confidence"] = result["confidence"]
        for class_name, confidence in result["all_confidences"]:
            row[f"prob_{class_name}"] = confidence

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

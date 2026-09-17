# Copyright CNRS 2024

"""Run DeepFaune on one image and save it if classified as mustelid.

Usage:
    python deepfaune/extractMustelidCrop.py IMAGE_PATH OUTPUT_DIRECTORY
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from predictTools import PredictorImage
from detectTools import cropSquareCVtoPIL


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def main():
    parser = argparse.ArgumentParser(
        description="Save a single image crop if DeepFaune classifies it as mustelid."
    )
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for the crop and CSV manifest",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--maxlag", type=float, default=20)
    parser.add_argument("--detector", default="DFbsMDS")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.image.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error(f"Unsupported image extension: {args.image.suffix}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = PredictorImage(
        filenames=[str(args.image)],
        threshold=args.threshold,
        maxlag=args.maxlag,
        LANG="en",
        birdclassification=True,
        detectorname=args.detector,
        device=args.device,
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
    }

    if predicted_class == "mustelid":
        image = cv2.imread(filename)
        if image is None:
            raise RuntimeError(f"Could not read image: {filename}")

        crop = cropSquareCVtoPIL(image, box)
        crop_path = args.output_dir / f"{args.image.stem}_mustelid.jpg"
        crop.save(crop_path, format="JPEG")
        row["crop_image"] = str(crop_path)
        print(f"Saved mustelid crop to {crop_path}")
    else:
        print(f"No crop saved; DeepFaune predicted: {predicted_class}")

    manifest_path = args.output_dir / "mustelid_crop.csv"
    pd.DataFrame([row]).to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from src.models import classification as pw_classification


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
        description="Classify one image with the fine-tuned mustelid classifier."
    )
    parser.add_argument("image", type=Path, help="Path to the image to classify")
    parser.add_argument(
        "--img-id",
        default=None,
        help="Identifier stored in the result; defaults to the image filename",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the classifier checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for saving the prediction result",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    model = pw_classification.CustomWeights(
        weights=args.checkpoint,
        class_names=CLASS_NAMES,
        device=args.device,
    )

    result = model.single_image_classification(
        str(args.image),
        img_id=args.img_id or args.image.name,
    )

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved prediction to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
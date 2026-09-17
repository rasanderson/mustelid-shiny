"""One-off utility to shrink a DeepFaune ViT checkpoint by storing it in fp16.

Only the on-disk checkpoint is affected: classifTools.py explicitly upcasts
the state dict back to fp32 on load, so runtime compute is unchanged (fp16
compute on CPU is unreliable/slow, so it is deliberately avoided).

Usage:
    python convert_weights_fp16.py [--weights PATH] [--output PATH]
"""

import argparse
from pathlib import Path

import torch


DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent
    / "deepfaune"
    / "deepfaune-vit_large_patch16_dinov3.lvd1689m.pt"
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a DeepFaune ViT checkpoint's state dict to fp16."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to overwriting --weights in place",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"Checkpoint does not exist: {args.weights}")

    output_path = args.output or args.weights
    before_size = args.weights.stat().st_size

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {
        k: v.half() if torch.is_floating_point(v) else v for k, v in state_dict.items()
    }
    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = state_dict
    else:
        checkpoint = state_dict

    torch.save(checkpoint, output_path)

    after_size = output_path.stat().st_size
    print(f"{args.weights} -> {output_path}")
    print(f"{before_size / 1e9:.2f} GB -> {after_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

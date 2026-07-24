"""
Load a MegaDetector-Classifier PyTorch Lightning checkpoint into
the PlainResNetClassifier architecture used during training.

Checkpoint used in this example:
weights/Crop/Plain/Crop_Res18_plain_071824-0-epoch=00-valid_mac_acc=nan.ckpt
"""

import torch
from collections import OrderedDict

# Import the model architecture used during training
from src.models.plain_resnet import PlainResNetClassifier


# ---------------------------------------------------------------------
# Step 1: Recreate the trained model architecture
# ---------------------------------------------------------------------
# These values match the training configuration:
#   num_classes = 2
#   num_layers  = 18
#
# If you train a different model in future, change these accordingly.
# ---------------------------------------------------------------------

model = PlainResNetClassifier(
    num_cls=2,
    num_layers=18
)


# ---------------------------------------------------------------------
# Step 2: Load the Lightning checkpoint
# ---------------------------------------------------------------------

ckpt = torch.load(
    "weights/Crop/Plain/Crop_Res18_plain_071824-0-epoch=00-valid_mac_acc=nan.ckpt",
    map_location="cpu"
)


# ---------------------------------------------------------------------
# Step 3: Remove the 'net.' prefix added by the Lightning wrapper
# ---------------------------------------------------------------------
# Checkpoint keys look like:
#
#   net.feature.conv1.weight
#   net.feature.layer1.0.conv1.weight
#   net.classifier.weight
#
# The PlainResNetClassifier expects:
#
#   feature.conv1.weight
#   feature.layer1.0.conv1.weight
#   classifier.weight
# ---------------------------------------------------------------------

state_dict = OrderedDict()

for key, value in ckpt["state_dict"].items():
    if key.startswith("net."):
        state_dict[key[4:]] = value


# ---------------------------------------------------------------------
# Step 4: Load the trained weights into the model
# ---------------------------------------------------------------------

missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False
)

print("Missing keys:")
print(missing_keys)

print("\nUnexpected keys:")
print(unexpected_keys)


# ---------------------------------------------------------------------
# Step 5: Switch the model into inference mode
# ---------------------------------------------------------------------

model.eval()


# ---------------------------------------------------------------------
# Step 6: Verify the classifier head
# ---------------------------------------------------------------------
# For the anteater example this should be:
#
#   torch.Size([2, 512])
#
# indicating a binary classifier.
# ---------------------------------------------------------------------

print("\nClassifier weight shape:")
print(model.classifier.weight.shape)


# ---------------------------------------------------------------------
# Step 7: Save the reconstructed model weights (optional)
# ---------------------------------------------------------------------
# This creates a plain PyTorch state_dict without the Lightning metadata.
# Uncomment if required.
# ---------------------------------------------------------------------

# torch.save(model.state_dict(), "anteater_classifier_state_dict.pth")

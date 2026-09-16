from src.models import classification as pw_classification

model = pw_classification.CustomWeights(
    weights="weights/Crop/Plain/Crop_Res50_plain_071824-0-epoch=14-valid_mac_acc=81.92.ckpt",
    class_names=[
        "mink",
        "otter",
        "pinemarten",
        "polecat",
        "stoat",
        "weasel",
    ],
    device="cpu",  # use "cuda" if available but shinyapps only has CPU support
)

result = model.single_image_classification(
    "predict_images/polecat.jpg", # Where to find the image
    img_id="polecat.jpg",         # Label for the image
)

print(result)
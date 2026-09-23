# Shiny for Python app. Start manually from command line in mustelid-shiny
# environment with
# python -m shiny run --reload app.py
import asyncio
import base64
import io

import pandas as pd
from PIL import Image, ImageDraw
from shiny import App, reactive, render, req, ui

from process_mustelid import classify_mustelid_image, classify_result_category

RESULT_MESSAGES = {
    "no_animal": "No animal detected",
    "other": "Other species",
    "mustelid": "Mustelid found",
    "otter": "Otter found",
}
# Categories where DeepFaune found an animal worth boxing
BOXABLE_CATEGORIES = {"mustelid", "otter", "other"}
# Fixed row order for the species probability table
SPECIES_ROWS = [
    "Other spp",
    "Mustelid",
    "Otter",
    "Weasel",
    "Pine marten",
    "Stoat",
    "Mink",
    "Polecat",
]

app_ui = ui.page_fillable(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "image_upload",
                "Choose an image file",
                accept=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
                multiple=False
            ),
            ui.input_action_button("identify", "Identify", disabled=True, width="150px"),
            ui.output_data_frame("species_table"),
            width="25%",
        ),
        ui.output_ui("image_display"),
    )
)

def server(input, output, session):
    detection_box = reactive.value(None)  # (x1, y1, x2, y2) or None
    species_probs = reactive.value(None)  # dict of {row_label: probability} or None

    @reactive.extended_task
    async def run_identify(image_path: str) -> dict:
        return await asyncio.to_thread(classify_mustelid_image, image_path)

    @reactive.effect
    def _update_identify_button():
        ui.update_action_button("identify", disabled=input.image_upload() is None)

    @reactive.effect
    @reactive.event(input.image_upload)
    def _reset_box_on_upload():
        detection_box.set(None)
        species_probs.set(None)

    @reactive.effect
    @reactive.event(input.identify)
    def _start_identify():
        file_info = req(input.image_upload())
        run_identify(file_info[0]["datapath"])

    @reactive.effect
    def _show_identify_result():
        status = run_identify.status()
        if status == "success":
            row = run_identify.result()
            category = classify_result_category(row["deepfaune_prediction"])
            if category in BOXABLE_CATEGORIES:
                detection_box.set((row["x1"], row["y1"], row["x2"], row["y2"]))
            else:
                detection_box.set(None)
            species_probs.set({
                "Other spp": row["deepfaune_score"] if category == "other" else 0.0,
                "Mustelid": row["deepfaune_score"] if category == "mustelid" else 0.0,
                "Otter": row["deepfaune_score"] if category == "otter" else 0.0,
                "Weasel": row["prob_weasel"] * row["deepfaune_score"] if category == "mustelid" else 0.0,
                "Pine marten": row["prob_pinemarten"] * row["deepfaune_score"] if category == "mustelid" else 0.0,
                "Stoat": row["prob_stoat"] * row["deepfaune_score"] if category == "mustelid" else 0.0,
                "Mink": row["prob_mink"] * row["deepfaune_score"] if category == "mustelid" else 0.0,
                "Polecat": row["prob_polecat"] * row["deepfaune_score"] if category == "mustelid" else 0.0,
            })
            message = RESULT_MESSAGES[category]
            ui.modal_show(
                ui.modal(message, title="Identification result", easy_close=True, footer=ui.modal_button("OK"))
            )
        elif status == "error":
            try:
                run_identify.result()
            except Exception as e:
                ui.modal_show(
                    ui.modal(str(e), title="Identification failed", easy_close=True, footer=ui.modal_button("OK"))
                )

    @render.ui
    def image_display():
        file_info = input.image_upload()
        
        if file_info is None:
            return ui.p("No image uploaded yet. Please select an image file.")
        
        # Get the file path from the uploaded file
        file_path = file_info[0]["datapath"]
        box = detection_box.get()

        if box is None:
            # Read the image file and encode it as base64
            with open(file_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()

            # Determine the MIME type based on the file name
            file_name = file_info[0]["name"].lower()
            if file_name.endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif file_name.endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"  # default
        else:
            # Draw the detection box directly on the pixels so it scales with the image
            image = Image.open(file_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            line_width = max(3, round(min(image.size) * 0.006))
            draw.rectangle(box, outline="red", width=line_width)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_data = base64.b64encode(buffer.getvalue()).decode()
            mime_type = "image/jpeg"

        # Return an img tag with base64 encoded data, capped to 1/4 of the browser width
        return ui.img(
            src=f"data:{mime_type};base64,{image_data}",
            style="max-width: 25vw; width: 100%; height: auto; display: block; margin-left: auto;"
        )

    @render.data_frame
    def species_table():
        probs = species_probs.get()
        if probs is None:
            values = ["\u2014"] * len(SPECIES_ROWS)
        else:
            values = [round(probs[label], 3) for label in SPECIES_ROWS]
        df = pd.DataFrame({"Species": SPECIES_ROWS, "Probability": values})
        return render.DataGrid(df, width="100%")

app = App(app_ui, server)


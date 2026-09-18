# Shiny for Python app. Start manually from command line in mustelid-shiny
# environment with
# python -m shiny run --reload app.py
import asyncio
import base64

from shiny import App, reactive, render, req, ui

from process_mustelid import classify_mustelid_image, classify_result_category

RESULT_MESSAGES = {
    "no_animal": "No animal detected",
    "other": "Other species",
    "mustelid": "Mustelid found",
}

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
            width="25%",
        ),
        ui.output_ui("image_display"),
    )
)

def server(input, output, session):
    @reactive.extended_task
    async def run_identify(image_path: str) -> dict:
        return await asyncio.to_thread(classify_mustelid_image, image_path)

    @reactive.effect
    def _update_identify_button():
        ui.update_action_button("identify", disabled=input.image_upload() is None)

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
        
        # Return an img tag with base64 encoded data, capped to 1/4 of the browser width
        return ui.img(
            src=f"data:{mime_type};base64,{image_data}",
            style="max-width: 25vw; width: 100%; height: auto; display: block; margin-left: auto;"
        )

app = App(app_ui, server)


# Shiny for Python app. Start manually from command line in mustelid-shiny
# environment with
# python -m shiny run --reload app.py
import base64

from shiny import App, render, ui

app_ui = ui.page_fillable(
    ui.card(
        ui.card_header("Image Upload"),
        ui.input_file(
            "image_upload",
            "Choose an image file",
            accept=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
            multiple=False
        ),
        ui.output_ui("image_display")
    )
)

def server(input, output, session):
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
        
        # Return an img tag with base64 encoded data
        return ui.img(
            src=f"data:{mime_type};base64,{image_data}",
            style="max-width: 100%; height: auto;"
        )

app = App(app_ui, server)

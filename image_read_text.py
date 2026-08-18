# ------------------------------------------------------------------
# READ TEXT FROM A PHOTO (OCR via pytesseract)
# Different job from a classifier: this pulls the TEXT out of an
# image rather than sorting the image into categories. Useful for
# accessibility ideas (reading labels, signs, notes aloud).
#
# NEEDS: the Tesseract OCR PROGRAM installed at system level — not
# just `pip install pytesseract`, which is only the Python wrapper.
#   Codespaces: handled automatically by the devcontainer.
#   Windows:    winget install UB-Mannheim.TesseractOCR
#   Mac:        brew install tesseract
# After installing, fully restart VS Code so the PATH updates.
#
# NOTE: type="pil" — pytesseract wants a PIL image, not the numpy
# array Gradio sends by default.
#
# To make it yours:
#   - ADAPT what happens with the text once you have it — that's
#     where your design thinking goes
# ------------------------------------------------------------------

import gradio as gr
import pytesseract
import os

# If you get "tesseract is not installed or it's not in your PATH"
# even after installing it, uncomment ONE of the lines below to point
# pytesseract straight at the program. Which one depends on whether it
# installed for all users (admin) or just you (no admin):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")

def read_text(image):
    text = pytesseract.image_to_string(image)
    if text.strip() == "":
        # ADAPT: what should your app say when it finds nothing?
        return "No text found in this photo."
    # ADAPT: what should your app DO with the text? Just showing it
    # is a start — a useful response depends on who this is for.
    return text

gr.Interface(
    fn=read_text,
    # ADAPT: label your input for your audience
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Upload or photograph something with text:"),
    outputs="text"
).launch(show_error=True)
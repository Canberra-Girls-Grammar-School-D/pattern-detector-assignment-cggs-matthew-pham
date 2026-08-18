# ------------------------------------------------------------------
# TONE-OF-MESSAGE CHECKER (pretrained sentiment)
# Builds on: Tutorial 6
# Checks whether a message might read harsher than intended
# before you hit send.
#
# To make it yours:
#   - ADAPT the responses: who's checking their message, and what
#     do they actually need to hear?
#   - ADAPT the threshold: how sure should the model be before
#     your app makes a claim?
# ------------------------------------------------------------------

import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def check_tone(text):
    result = classifier(text)
    label = result[0]["label"]
    score = result[0]["score"]
    # ADAPT: threshold and all three messages
    if score < 0.7:
        return f"Hard to tell. It leans {label.lower()}, but only just ({score:.2f}). Maybe reword?"
    elif label == "NEGATIVE":
        return f"This might land harsher than you mean ({score:.2f} confidence)."
    else:
        return f"Reads fine ({score:.2f} confidence)."

gr.Interface(
    fn=check_tone,
    # ADAPT: label your input for your audience
    inputs=gr.Textbox(label="Paste your message:"),
    outputs="text"
).launch(show_error=True)
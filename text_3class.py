# ------------------------------------------------------------------
# 3-CLASS SENTIMENT (adds "neutral")
# Builds on: Tutorial 6 — ONE changed line (the model= argument)
# Trained on tweets, so a better fit for casual/social text.
#
# GOTCHA: this model's labels are lowercase ("positive", "negative",
# "neutral") — Tutorial 6's default model used UPPERCASE. Any
# if label == "POSITIVE" checks must change to match.
#
# To make it yours:
#   - ADAPT the responses for all THREE labels
#   - ADAPT the threshold
# ------------------------------------------------------------------

import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def check_sentiment(text):
    result = classifier(text)
    label = result[0]["label"]
    score = result[0]["score"]
    # ADAPT: threshold and messages. Note the unsure branch still
    # tells the user the best guess — saying nothing is less useful
    # than an honest "probably X, but I'm not certain". — note lowercase labels
    if score < 0.6:
        return f"Leaning {label}, but not confidently ({score:.2f})."
    elif label == "positive":
        return f"Positive ({score:.2f} confidence)"
    elif label == "neutral":
        return f"Neutral — no strong feeling either way ({score:.2f} confidence)"
    else:
        return f"Negative ({score:.2f} confidence)"

gr.Interface(
    fn=check_sentiment,
    # ADAPT: label your input for your audience
    inputs=gr.Textbox(label="Say something:"),
    outputs="text"
).launch(show_error=True)
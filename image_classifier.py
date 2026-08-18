# ------------------------------------------------------------------
# IMAGE CLASSIFIER (pretrained MobileNet)
# Builds on: Tutorial 7
# Knows ~1000 everyday objects out of the box. No training needed.
#
# To make it yours:
#   - ADAPT the response messages and threshold below
#   - ADAPT the interface labels for your audience
# ------------------------------------------------------------------

import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def prepare_image(image):
    image = tf.image.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def classify(image):
    prepared = prepare_image(image)
    predictions = model.predict(prepared)
    results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    top_guess = results[0][0]
    label = top_guess[1]
    score = top_guess[2]
    # ADAPT: your threshold — how confident is confident enough for YOUR app?
    if score < 0.5:
        # ADAPT: what should your app say when it's not sure?
        return f"My best guess is {label}, but I'm not confident ({score:.2f})."
    else:
        # ADAPT: what would actually help your audience, beyond the label?
        return f"That looks like {label} ({score:.2f} confidence)"

gr.Interface(
    fn=classify,
    # ADAPT: label your input for your audience
    inputs=gr.Image(sources=["upload", "webcam"], label="Upload or take a photo:"),
    outputs="text"
).launch(show_error=True)

# WANT LIVE VIDEO INSTEAD, like Teachable Machine's preview?
# Swap the whole gr.Interface block above for this one. It classifies
# continuously as the camera runs, instead of one photo at a time.
# Heads up: it runs the model on every frame, so it can be slow —
# test it before relying on it for a demo.
#
# gr.Interface(
#     fn=classify,
#     inputs=gr.Image(sources=["webcam"], streaming=True, label="Point the camera:"),
#     outputs="text",
#     live=True
# ).launch(show_error=True)
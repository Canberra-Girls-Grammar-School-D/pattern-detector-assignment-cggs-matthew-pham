# ------------------------------------------------------------------
# IMAGE CLASSIFIER (your own Teachable Machine model)
# Builds on: Tutorial 8
# Runs with the included sample_image_model.h5 so you can see it
# work, then you replace that with YOUR export.
#
# To make it yours:
#   1. Train an Image Project at teachablemachine.withgoogle.com
#   2. Export: Tensorflow tab -> Keras -> Download, unzip
#   3. Drag YOUR keras_model.h5 into this folder (replace the sample)
#   4. ADAPT class_names to match YOUR classes, in the same order
#   5. ADAPT the responses and threshold
#
# TIP: if you trained with the webcam, test with the webcam too.
# A model trained in one spot often does worse on photos taken
# somewhere else, because it partly learned the background.
# ------------------------------------------------------------------

import gradio as gr
import tf_keras
import numpy as np
import os
from PIL import Image, ImageOps

# Look for the model file next to this script, so it works no matter
# which folder you run from
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_image_model.h5")

# Teachable Machine exports were built with Keras 2, so we load them
# with tf_keras (the Keras 2 library) rather than tf.keras (Keras 3),
# which rejects part of the file
# compile=False because we're only using the model to predict, not
# to train it further — this also avoids a harmless warning
model = tf_keras.models.load_model(MODEL_FILE, compile=False)

# ADAPT: your classes, in the exact order Teachable Machine listed them.
# Get the order wrong and every answer is confidently mislabelled.
class_names = ["Bottle", "Honey"]

def prepare_image(image):
    # Teachable Machine crops to a square from the centre before
    # shrinking, so we do the same. Plain resizing squashes the photo
    # and the model then sees something it was never trained on.
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image = np.asarray(image).astype(np.float32)
    # Scale 0-255 colour values into the -1 to 1 range TM trained on
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=0)
    return image

def classify(image):
    prepared = prepare_image(image)
    prediction = model.predict(prepared)
    index = np.argmax(prediction)
    label = class_names[index]
    score = prediction[0][index]
    # ADAPT: your threshold and messages
    if score < 0.8:
        return f"Might be {label}, but I'm not confident ({score:.2f})."
    else:
        return f"{label} ({score:.2f} confidence)"

gr.Interface(
    fn=classify,
    # ADAPT: label your input for your audience
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Upload or take a photo:"),
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
#     inputs=gr.Image(type="pil", sources=["webcam"], streaming=True, label="Point the camera:"),
#     outputs="text",
#     live=True
# ).launch(show_error=True)
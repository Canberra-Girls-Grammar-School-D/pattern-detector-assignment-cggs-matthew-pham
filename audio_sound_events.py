# ------------------------------------------------------------------
# SOUND EVENT IDENTIFIER (pretrained general sound classification)
# Builds on: Tutorial 6's shape, different pipeline model.
#
# This recognises general sound EVENTS (music, speech, animal
# sounds, environmental noise) rather than spoken commands. Good fit
# for: bird call identifiers, "what's that sound" apps, ambient
# monitoring. NOT a fit for voice commands — see
# audio_keyword_spotter.py for that.
#
# The exact classes depend on the model's training set — check
# MIT/ast-finetuned-audioset on Hugging Face for the full list this
# one recognises (it's trained on AudioSet's ~500+ categories).
#
# AUDIO FORMAT: .wav files work everywhere with no extra setup.
# MP3, FLAC and M4A need ffmpeg installed (the Codespace devcontainer
# handles this; locally, run `winget install ffmpeg` on Windows).
# If in doubt, use .wav — that's what the test_files folder provides.
# Recordings at any sample rate get resampled to the 16kHz these
# models expect, which is what torchaudio in requirements.txt is for.
#
# NOTE: type="numpy" means Gradio hands over (sample_rate, array)
# directly and we convert to the float32 the pipeline wants.
# Microphone input depends on browser permissions, so test it before
# relying on it.
#
# To make it yours:
#   - ADAPT the responses and threshold
#   - ADAPT the interface labels
# ------------------------------------------------------------------

import gradio as gr
from transformers import pipeline

classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")

def identify_sound(audio):
    sample_rate, audio_array = audio
    # If the recording is stereo (2 channels), average down to mono —
    # the pipeline only accepts single-channel audio
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    # Gradio gives int16 audio; the pipeline expects float32 in [-1, 1]
    audio_float = audio_array.astype("float32") / 32768.0
    results = classifier({"sampling_rate": sample_rate, "raw": audio_float})
    top = results[0]
    label = top["label"]
    score = top["score"]
    # ADAPT: threshold and messages. Note the unsure branch still
    # tells the user the best guess — saying nothing is less useful
    # than an honest "probably X, but I'm not certain".
    if score < 0.3:
        return f"Might be {label}, but I'm not sure ({score:.2f})."
    else:
        return f"Sounds like: {label} ({score:.2f} confidence)"

gr.Interface(
    fn=identify_sound,
    # ADAPT: label your input for your audience
    inputs=gr.Audio(type="numpy", label="Upload a sound:"),
    outputs="text"
).launch(show_error=True)
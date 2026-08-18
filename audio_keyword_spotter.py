# ------------------------------------------------------------------
# VOICE COMMAND RECOGNIZER (pretrained keyword spotting)
# Builds on: Tutorial 6 — same library, same list-of-dicts result
# shape, just a different pipeline name and an audio input.
#
# This model recognises a small set of SPOKEN COMMAND WORDS, not
# general sounds. It was trained on: yes, no, up, down, left, right,
# on, off, stop, go, plus silence and "unknown" for anything else.
# Good fit for: voice-controlled apps, accessibility switches,
# simple spoken commands. NOT a fit for: identifying sounds like
# barks, sirens, or music — see audio_sound_events.py for that.
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

classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")

def recognize_command(audio):
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
    if label == "_unknown_" or score < 0.5:
        return f"Didn't catch that clearly. Closest match was {label} ({score:.2f})."
    else:
        return f"Heard: {label} ({score:.2f} confidence)"

gr.Interface(
    fn=recognize_command,
    # ADAPT: label your input for your audience
    inputs=gr.Audio(type="numpy", label="Say a command:"),
    outputs="text"
).launch(show_error=True)
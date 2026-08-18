# Project Options Guide

A reference for Pattern Detector App proposals: contexts, what's realistically in scope, and the tools and libraries available at this level. Scoped for Y9/10 with Gradio and light Python, not the Y11/12 Streamlit/Jupyter toolset — everything here should be buildable with what the tutorials have already taught, or a small, honest step beyond it.

## Contexts and example ideas

Each context from the task sheet, with concrete starting ideas. None of these need to be copied exactly — they're here to show the shape of a good idea, not a menu to pick from unchanged.

**🌿 Environment and Nature**
- Bird call identifier for a nature reserve visitor
- Plant disease spotter for a home gardener
- Cloud type or weather pattern identifier
- Water quality / pollution photo checker

**🎨 Creative and Performance**
- Instrument tuning or pitch practice coach
- Art style identifier (impressionist vs pop art vs...)
- Vocal warm-up feedback tool

**🏃 Sport and Movement**
- Action recognition from photos (is this a serve or a smash?)
- Equipment or technique identification
- Movement counting from images

**♿ Daily Life and Accessibility**
- Sign language letter recognizer (from hand photos)
- Recycling sorter (what bin does this go in)
- Tone-of-message checker (does this sound rude or joking)
- Reading a photographed note, label or sign aloud

**Or propose your own.** The "deserves automating" test is the one that matters most here: does this let someone do something a human judge couldn't — because the expert isn't there, the scale is too big for a person, consistency matters, or there's no one around to ask? If a friend standing right there could just tell them the answer for free, it's probably not the strongest fit.

## What's in scope

This unit is **classification only**: sorting an input into one of a small number of categories (2–5 classes is the sweet spot). That covers almost everything above.

**Out of scope for this unit** (name these if a student asks, don't build them): regression (predicting a number, like a house price or exam score), clustering (grouping things with no predefined categories), recommendation systems, and reinforcement learning. These are real ML techniques, just not ones the tools here support without moving to a very different toolkit (they show up at Y11/12 level with Streamlit and libraries like scikit-learn).

## Choosing an approach

Three roads, not mutually exclusive:

| Approach | When it fits | What it costs |
|---|---|---|
| **Pretrained model** | A general-purpose model plausibly already knows this (most objects, most text sentiment, many sounds) | Almost nothing — a few lines, no training wait, no data collection |
| **Train your own (Teachable Machine)** | Nothing pretrained fits the specific thing being classified — a group's own slang, specific local packaging, a narrow category no general model knows | Real time: collecting 30+ varied examples per class, training, and testing honestly for the failures that come from thin data |
| **Cloud API** | A specific, well-defined task an API is built for (sentiment, emotion) and internet access is safe to assume | An account, a key, and handling that key properly (`.env`, never committed) |

**Default to trying pretrained first for image, text, and most audio ideas** — it's the fastest way to see if a working starting point already exists. Train-your-own becomes the right call specifically when nothing pretrained covers the thing being classified, not as a fallback for when pretrained "isn't good enough."

**A note on pose projects.** Teachable Machine has a Pose Project mode, and it's genuinely good — it reduces a person to skeleton keypoints, so background and clothing stop mattering. But pose models only export to TensorFlow.js, not the Keras format Python needs, so they can't be used with the Gradio workflow this unit is built on. Body-position ideas can still work as **Image Projects** (training on photos rather than skeletons), just with the usual image-model sensitivity to background and lighting.

## Libraries and tools by input type

Everything in **bold** is already built into a tutorial. Everything else is a genuine, accessible extension, verified to work with a small amount of extra code — but not pre-tested with students, so try it yourself first.

### Image

| Tool | What it does | Example output |
|---|---|---|
| **`tf.keras.applications.MobileNetV2`** (Tutorial 7) | Pretrained, ~1000 everyday object classes | `golden_retriever (0.87)` |
| **Teachable Machine, Image Project** (Tutorial 8) | Train your own from photos or webcam | Whatever classes you define |
| `pytesseract` | Reads text out of a photo (OCR) | A photo of a handwritten note → the words as text |

`pytesseract` note: genuinely simple in code (`pytesseract.image_to_string(image)`, one line), but needs the Tesseract program installed at the system level, not just `pip install` — a small extra setup step, worth testing in your own Codespace before offering it to a student.

### Text

| Tool | What it does | Example output |
|---|---|---|
| **`transformers` `pipeline("sentiment-analysis")`** (Tutorial 6) | Pretrained, positive/negative | `POSITIVE (0.98)` |
| `pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")` | Same library, 3-class (adds neutral), trained on tweets — better fit for slang/social content | `neutral (0.71)` |
| **Twinword via RapidAPI** (Tutorial 9) | Cloud API, 3-class sentiment, no local model needed | `type: negative, score: -0.31` |

### Audio

| Tool | What it does | Example output |
|---|---|---|
| **Teachable Machine, Audio Project** | Train your own from short recordings | Whatever sound classes you define |
| YAMNet (TensorFlow Hub) | Pretrained, 521 general sound event classes (barking, sirens, music...) | `Dog bark (0.91)` |
| `transformers` `pipeline("audio-classification")` | Same `transformers` library students already know, different pipeline name | List-of-dict, same shape as text sentiment |

Audio input in Gradio (`gr.Audio(sources=["microphone"])`) depends on browser microphone permission — test this yourself in a real Codespace before a student relies on it live; file upload is the safe fallback.

## Interface tool

**Gradio** is the one interface tool this unit uses, and it's sufficient for every idea above — image, text, and audio all have a matching Gradio input type. Streamlit and Jupyter Notebooks are Y11/12 tools; there's no reason to introduce them here.

## A few honest cautions

- **Public datasets** (downloading someone else's data rather than collecting your own) are possible but add real extra work — file formats to handle, cleaning, matching what a model expects. Worth mentioning as an option for a strong student chasing an enhancement, not a default path.
- **New libraries beyond what's taught** need adding to `requirements.txt` and testing in a real Codespace before a lesson — treat every entry above the same way `07`'s TensorFlow pin was handled: verify it installs and runs before a student depends on it.
- **Anything using a webcam or microphone** needs a permissions dry run. Anything using an API key needs the `.env`/`.gitignore` pattern from Tutorial 9, even if the specific API is different from Twinword.

## Starter code

Two kinds below: **extension tool snippets** (tools no tutorial covers) and **domain skeletons** (a bare starting shape for common project types, built from tutorial code students already have). Verification status is marked on each — ✅ means tested this session or directly from a tutorial's dry-run-tested code; ⚠️ means adapted from documentation and needs one run in a real Codespace before handing to a student.

### Extension tools

**Reading text from a photo (`pytesseract`)** ⚠️ — needs `pytesseract` in `requirements.txt` AND the Tesseract program itself (`sudo apt-get install tesseract-ocr` in the Codespace terminal, or added to the devcontainer):

```python
import gradio as gr
import pytesseract

def read_text(image):
    text = pytesseract.image_to_string(image)
    if text.strip() == "":
        return "No text found in this photo"
    return text

gr.Interface(
    fn=read_text,
    inputs=gr.Image(label="Upload a photo with text in it:", type="pil"),
    outputs="text"
).launch(show_error=True)
```

Note `type="pil"` on the image input — pytesseract wants a PIL image, not the numpy array Gradio sends by default.

**Sound classification, pretrained (`transformers` audio pipeline)** ⚠️ — same library and same list-of-dict output shape as Tutorial 6, just a different pipeline name:

```python
import gradio as gr
from transformers import pipeline

classifier = pipeline("audio-classification")

def classify_sound(audio_file):
    results = classifier(audio_file)
    top = results[0]
    return f"{top['label']} ({top['score']:.2f})"

gr.Interface(
    fn=classify_sound,
    inputs=gr.Audio(label="Upload a sound:", type="filepath"),
    outputs="text"
).launch(show_error=True)
```

Note `type="filepath"` — the pipeline wants a file path, not raw audio data. Microphone input (`sources=["microphone"]`) needs a browser-permission dry run first; file upload is the safe fallback.

**3-class sentiment (adds neutral)** ⚠️ — one changed line from Tutorial 6, everything else identical:

```python
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

Labels come back lowercase (`positive`/`negative`/`neutral`) from this model, so any `if label == "POSITIVE"` checks from Tutorial 6 need their capitalisation adjusted — a genuine gotcha worth warning students about.

**YAMNet (521 sound events, TensorFlow Hub)** ⚠️ — needs `tensorflow_hub` in `requirements.txt`; more wiring than the audio pipeline above, so prefer the pipeline unless a student specifically needs YAMNet's class list. Not included here as a full snippet for that reason — if a student reaches this level, build it with them from the TensorFlow Hub YAMNet tutorial.

### Domain skeletons

Each of these is tutorial code with the blanks renamed — the point is showing how little changes between "the tutorial" and "my project."

**Recycling sorter** ✅ (Tutorial 8's exact structure, different classes) — train in Teachable Machine with classes like `Recycling`, `Landfill`, `Compost` on photos of actual local packaging, export, then Tutorial 8's code with:

```python
class_names = ["Recycling", "Landfill", "Compost"]
```

and responses that tell the user what to *do*, not just the label:

```python
    if score < 0.6:
        return "I'm not sure — check the packaging symbol or ask"
    else:
        return f"That goes in {label} ({score:.2f} confidence)"
```

**Tone-of-message checker** ✅ (Tutorial 6's exact structure, different responses) — the sentiment pipeline unchanged, with response messages written for the actual use case:

```python
    if score < 0.7:
        return "Hard to tell — tone is genuinely ambiguous here, maybe reword?"
    elif label == "NEGATIVE":
        return f"This might land harsher than you mean ({score:.2f} confidence)"
    else:
        return f"Reads fine ({score:.2f} confidence)"
```

**Bird call identifier** ⚠️ (Teachable Machine Audio Project) — same train/export/load workflow as Tutorial 8 but with an audio model; the export format differs from the image `.h5`, so this one specifically needs a teacher dry run of the export-and-load step before a student commits to it.

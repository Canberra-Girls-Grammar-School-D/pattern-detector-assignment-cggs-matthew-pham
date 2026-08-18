# Test files

Files for verifying each starter runs, before adapting it.

## Included here

| File | For | What to expect |
|---|---|---|
| `ocr_clean_note.png` | `image_read_text.py` | Reads all three lines accurately |
| `ocr_faded_label.png` | `image_read_text.py` | Harder: low contrast — reads, but shows OCR working near its limits |
| `ocr_no_text.png` | `image_read_text.py` | Triggers the "no text found" branch |
| `audio_test_tone.wav` | `audio_keyword_spotter.py` or `audio_sound_events.py` | Plumbing test only: a pure tone, not a real word or sound. Confirms the code runs; won't classify meaningfully in either starter |

## Audio format note

**Prefer `.wav` for audio test files.** MP3, FLAC and M4A all need `ffmpeg` installed to decode — the Codespace devcontainer installs it, but a local machine may not have it (`winget install ffmpeg` on Windows). WAV needs nothing extra. ESC-50 and the Speech Commands dataset both provide WAV; Freesound and Pixabay often default to MP3, so check the format before downloading.

## Not included — grab or make these yourself

- **Photos of everyday objects** (`image_classifier.py`): photograph 3–4 things on your desk — a mug, a keyboard, a plant. MobileNet knows common objects, so ordinary photos work. Include one weird angle or partial view to test the "not sure" branch.
- **For `audio_keyword_spotter.py`** (needs spoken command words, not general sounds): record yourself saying "yes," "no," "stop," or "go" on your phone — a couple of seconds each, clear speech. The **Google Speech Commands dataset** (the actual dataset this model was trained on) is the ideal source if you want a proper set: search "speech commands dataset download," or `tensorflow_datasets` includes it directly.
- **For `audio_sound_events.py`** (general sound events): **ESC-50** (github.com/karolpiczak/ESC-50) is the right fit here — 2000 labelled clips across 50 categories (dog bark, clapping, rain, siren), CC-BY licensed, individual `.wav` files on GitHub. Freesound.org works too if you want sounds outside ESC-50's 50 categories.
- **`sample_image_model.h5`** (`image_trained_yours.py`): train a trivial 2-class Teachable Machine **Image Project** (two objects from your desk, 30 samples each), export as Keras, rename, and place it in the repo root. This doubles as the dry-run verifying TM exports load via `tf_keras`.
- **Messages to test tone with** (`text_tone_checker.py`, `text_3class.py`): no files needed — type directly. Try one clearly positive, one blunt/harsh, one genuinely ambiguous, and something with current slang.
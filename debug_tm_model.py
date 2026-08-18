# ------------------------------------------------------------------
# DIAGNOSTIC SCRIPT — not a starter, a debugging tool.
#
# Run this when a Teachable Machine model gives wrong or identical
# answers. It prints what the model expects, what it's being given,
# and what it returns, so you can see exactly where things go wrong.
#
# Usage:
#   python debug_tm_model.py sample_image_model.h5 photo1.jpg photo2.jpg
# ------------------------------------------------------------------

import sys
import os
import numpy as np
import tf_keras
from PIL import Image, ImageOps

if len(sys.argv) < 3:
    print("Usage: python debug_tm_model.py <model.h5> <image1> [image2] ...")
    raise SystemExit(1)

model_path = sys.argv[1]
image_paths = sys.argv[2:]

print("=" * 60)
print("MODEL")
print("=" * 60)
model = tf_keras.models.load_model(model_path, compile=False)
print("File:            ", model_path)
print("Expects input:   ", model.input_shape)
print("Produces output: ", model.output_shape)
print("Number of classes:", model.output_shape[-1])
print()
print("If 'Expects input' is NOT something like (None, 224, 224, 3),")
print("this model does not take images and this script won't help.")
print()

for path in image_paths:
    print("=" * 60)
    print("IMAGE:", path)
    print("=" * 60)

    if not os.path.exists(path):
        print("  FILE NOT FOUND — skipping")
        continue

    raw = Image.open(path).convert("RGB")
    print("Original size:   ", raw.size, "| mode:", raw.mode)

    fitted = ImageOps.fit(raw, (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(fitted).astype(np.float32)
    print("After crop+fit:  ", arr.shape, "| range: %.1f to %.1f" % (arr.min(), arr.max()))

    scaled = (arr / 127.5) - 1
    print("After scaling:   ", scaled.shape, "| range: %.2f to %.2f" % (scaled.min(), scaled.max()))

    batched = np.expand_dims(scaled, axis=0)
    print("After batching:  ", batched.shape)

    prediction = model.predict(batched, verbose=0)
    print()
    print("RAW PREDICTION:  ", prediction)
    print("Winning index:   ", int(np.argmax(prediction)))
    print("Winning score:   ", float(np.max(prediction)))
    print()

print("=" * 60)
print("WHAT TO LOOK FOR")
print("=" * 60)
print("- Different images giving DIFFERENT raw predictions = model works,")
print("  the problem is elsewhere (class_names order, or the app itself)")
print("- Different images giving IDENTICAL raw predictions = the model")
print("  isn't distinguishing them; retrain with more varied examples")
print("- 'Expects input' not matching (None, 224, 224, 3) = wrong model")
print("  type for this code (a Pose project exports a different shape)")
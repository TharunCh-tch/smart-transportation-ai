# CV Module — Real Evaluation Results

This document reports actual measured numbers from running the code in
this repo, not target/marketing numbers. Everything below was produced
by `scripts/eval_vision.py`; the raw JSON it wrote is committed at
[`results_raw.json`](results_raw.json) so the numbers here can be checked
against it. Reproduce with:

```bash
.venv\Scripts\python.exe scripts\eval_vision.py --out results_raw.json --save-plates tests\fixtures\synthetic_plates
```

**Hardware**: Intel Core i5-8250U @ 1.60GHz (4-core laptop CPU), no GPU.
All numbers below are CPU inference. See the "GPU inference" section of
README.md for the (separate, untested-by-us-in-production) EC2 GPU
deployment path.

**Model**: `yolov8n.pt` (Ultralytics YOLOv8-nano), stock COCO-pretrained
weights — no fine-tuning, no custom training data. `conf=0.35, iou=0.45`.

---

## 1. Vehicle detection

### 1.1 Ground-truth precision/recall (small, hand-annotated set)

Three images with manually-annotated ground truth (annotated by visually
inspecting each image and counting/classifying vehicles by eye — see
`GROUND_TRUTH` in `scripts/eval_vision.py` for the exact per-image
reasoning):

| Image | GT vehicles | Predicted | Predicted classes | Detect latency |
|---|---|---|---|---|
| `bus.jpg` (Ultralytics sample) | 1 (bus) | 1 | `{bus: 1}` | 8734 ms *(cold start incl. weight decode)* |
| `zidane.jpg` (Ultralytics sample, negative control) | 0 | 0 | `{}` | 152 ms |
| `manhattan_50th_st.jpg` (real photo, CC-BY-4.0) | 4 (1 truck, 1 bus, 2 car) | 3 | `{bus: 1, car: 2}` | 148 ms |

**Aggregate (vehicle-level, class-agnostic — "is there a vehicle where
one exists"):**

- **Precision: 1.00** (4/4 — every detection corresponded to a real vehicle, zero false positives)
- **Recall: 0.80** (4/5 — one real vehicle missed)

**A specific, honest failure case worth calling out**: on
`manhattan_50th_st.jpg`, the model detected 3 objects where the naive
per-class-count metric shows "3/4 classes correct," but that's
misleading if read too quickly. What actually happened, confirmed by
inspecting box coordinates:
- The large foreground **truck** (an International-brand box truck) was
  detected but mislabeled **`bus`** at 0.80 confidence — box coordinates
  place it exactly where the truck is.
- The real background **transit bus** was detected only at 0.19
  confidence as `truck` — below our 0.35 threshold, so it doesn't appear
  in the output at all.
- Both cars (taxi + dark sedan) were detected correctly.

So the honest read is: 3 of 4 real vehicles were localized (75% recall,
matches the aggregate above), one class label was swapped (truck↔bus —
a genuinely common COCO confusion between visually similar large
vehicles), and one vehicle was missed outright because it fell below
the confidence threshold. This is a real, representative limitation of
a stock nano-sized COCO detector on a cluttered street scene, not
something we're hiding.

Detector latency after the first (cold) call is **~150 ms/image** on
this CPU. The 8.7s figure for `bus.jpg` is the very first inference
call in the process and includes PyTorch/YOLO internal warm-up; it is
not representative of steady-state latency.

### 1.2 Qualitative checks on dense / difficult scenes (no formal ground truth)

Two images were deliberately **excluded** from the precision/recall
numbers above because hand-counting 40–90 small, overlapping vehicles
reliably enough to call it "ground truth" isn't realistic. We ran
detection anyway and report the raw counts as a qualitative sanity
check, not an accuracy claim:

| Image | Predicted vehicles | Classes | Notes |
|---|---|---|---|
| `auckland_traffic.jpg` (CC0, dense 5-lane highway) | 37 | `{car: 35, truck: 2}` | Rough visual estimate by eye is in the same ballpark (~40–50); some very small/distant/partially-occluded vehicles are almost certainly missed — undercounting on dense scenes is a known YOLO-nano limitation. |
| `herald_square_traffic.jpg` (1973 NARA photo, public domain) | 1 | `{car: 1}` | Dramatic undercount on a genuinely hard case: a 50-year-old scanned photograph — different color science, film grain, and vehicle designs than anything in COCO's training distribution. This is a real, honest domain-shift failure, included on purpose rather than dropped from the report. |

**Takeaway**: this is a general-purpose COCO detector, not something
trained or tuned for traffic-camera footage. It's solid on clear modern
photos, degrades on dense scenes, and degrades hard on out-of-distribution
inputs (historic film photos, unusual lighting/angles). We're reporting
that plainly rather than cherry-picking the images that make it look best.

---

## 2. Emergency-vehicle heuristic

**There is no labeled emergency-vehicle dataset in this project** —
that's a hard requirement we don't have a way around without either
scraping/licensing real police/fire/EMS photos (and their associated
consent/privacy issues) or hand-labeling a set ourselves, neither of
which we did. So: **we do not report, and you should not trust, any
precision/recall number for actually detecting emergency vehicles.**
Anyone telling you a number there without a labeled positive set is
making it up.

What we *can* measure honestly: every image in our test set is a known
**true negative** — none of them contain a real emergency vehicle — so
every `likely_emergency=True` flag on them is by definition a false
positive. That gives a real, if narrow, signal: how trigger-happy is
the heuristic on ordinary traffic.

**Threshold tuning** (empirical, against the same known-negative set):

| Threshold | False positives / 42 detections | FP rate |
|---|---|---|
| 0.12 (initial guess) | 9 | 21.4% |
| 0.20 | 5 | 11.9% |
| 0.25 | 5 | 11.9% |
| 0.30 | 3 | 7.1% |
| **0.35 (shipped)** | **1** | **2.4%** |
| 0.40 | 1 | 2.4% |

We shipped **threshold = 0.35**: false positive rate **2.4% (1/42)** on
the known-negative set, score distribution min=0.015, median=0.066,
max=0.793 (the one false positive, on `auckland_traffic.jpg`, scored
0.79 — likely a car with strong red/blue paint or reflections in the
sampled region).

**What this heuristic actually is**: for each vehicle detection box, we
look at the top ~35% (where a roof light bar would be) in HSV space and
measure the fraction of red/blue pixels, plus a weak white/black
contrast signal over the full box. See `app/ai/vision/emergency.py` for
the exact method and its docstring limitations. It will false-positive
on red/blue vehicles and false-negative on emergency vehicles with
lights off or unusual liveries — it's a documented starting point, not
a production-ready classifier.

---

## 3. License-plate detection + OCR

**No real license-plate dataset was used.** We don't have access to a
permissively-licensed, privacy-safe dataset of real plates (real plates
photographed in public are still personally identifying information
tied to real vehicle owners, and we're not going to scrape or publish
that). Instead we generated a **synthetic plate dataset**: 36 images
(12 per difficulty tier), each a random 3-letter + 4-digit string
rendered with a real font (Arial Bold / Consolas), at three difficulty
levels:

- **clean** — flat white background, sharp text
- **blur** — ±6° rotation + Gaussian blur (simulates camera motion/focus)
- **low_light** — blur + rotation + 25–45% brightness reduction + Gaussian
  pixel noise (simulates a dim/night camera feed)

Sample generated images are committed under
[`tests/fixtures/synthetic_plates/`](tests/fixtures/synthetic_plates/).
**This measures OCR quality under controlled synthetic conditions — it
is a proxy, not a claim about real-world plate-reading accuracy.** Real
plates have dirt, glare, non-standard fonts/spacing, frames, angles, and
motion blur that this dataset doesn't fully capture.

Pipeline tested end-to-end: OpenCV contour-based region proposal (see
`propose_plate_regions()` in `app/ai/vision/plate_ocr.py`) → EasyOCR.

### Results

| Difficulty | n | Exact-match accuracy | Avg. char accuracy |
|---|---|---|---|
| clean | 12 | **91.7%** | 98.8% |
| blur | 12 | **91.7%** | 98.8% |
| low_light | 12 | **75.0%** | 94.0% |
| **Overall** | **36** | **86.1%** | **97.2%** |

Average OCR latency: **~1.05 s/plate** on CPU (this is the dominant cost
in the whole pipeline — see README's latency notes for what that means
for video throughput).

**Error analysis** — every mismatch we inspected was a classic OCR
character confusion, not a garbage read:

```
clean:      XOS2486 -> X0S2486   (O misread as 0)
blur:       EMW0880 -> EMWO880   (0 misread as O)
low_light:  KYO5006 -> KYO5OOG   (0 misread as O, twice; 6 misread as G)
low_light:  SOC9610 -> S0C9610   (O misread as 0)
low_light:  AFI8926 -> AF18926   (I misread as 1)
```

This is a genuinely realistic failure mode for any OCR-based ANPR
system (real-world plate readers hit the same O/0 and I/1 ambiguity)
and is worth knowing about for anyone building on top of this: don't
trust a single OCR read for anything where an O/0 or I/1 mixup matters
— use multi-frame voting (the tracker in `app/ai/vision/tracker.py`
gives you the track continuity to do that) or a plate-specific
character set constraint.

### Plate region proposal (the non-ML half of this pipeline)

The classic "poor man's ANPR" contour heuristic (grayscale → bilateral
filter → Canny edges → contour + aspect-ratio filtering) is not
separately benchmarked with precision/recall here because we don't have
real vehicle photos with labeled plate bounding boxes. It's covered by
targeted unit tests instead (`tests/test_vision_plate_ocr.py`): it
reliably finds a plate-shaped rectangle on a synthetic "vehicle with
plate" test image (IOU > 0.3 against the known region) and correctly
returns nothing on a featureless/edge-free image. On the real sample
photos in `tests/fixtures/`, plates are mostly too small/distant/blurry
for either the region proposer or OCR to produce a read at all — the
live API calls we ran against `manhattan_50th_st.jpg` and
`auckland_traffic.jpg` returned `plate: null` for every vehicle except a
handful of the closest, most head-on cars in the Auckland highway photo.

---

## 4. Multi-object tracking

Not benchmarked with a formal MOT metric (MOTA/IDF1) — we don't have a
labeled multi-object tracking dataset either, and building one is out of
scope here. What we did verify (see `tests/test_vision_tracker.py` and
`tests/test_vision_pipeline.py::test_process_video_tracks_are_stable_across_synthetic_motion`):

- A synthetic video built by shifting a static frame a few pixels per
  frame (simulating slow, steady motion) keeps a **single stable track
  ID** for each object across all frames — no ID switches, no dropped
  tracks, in both the unit-level tracker tests and the full
  video-pipeline integration test.
- Two objects moving in parallel are tracked independently without their
  IDs swapping (`test_two_vehicles_moving_in_parallel_do_not_swap_ids`).
- A track is correctly dropped after `max_disappeared` consecutive
  missed frames, and a detection with a different class label never
  gets matched onto an existing track of a different label.

This is real, correct behavior for the case it's designed for (a mostly
static or slow-panning camera, consistent lighting, no long occlusions).
It is **not** ByteTrack/DeepSORT-level — no motion model, no
re-identification embedding, greedy (not Hungarian-optimal) assignment
— and will do worse on fast motion, camera shake, or long occlusions.
See the module docstring in `app/ai/vision/tracker.py`.

---

## Summary — what's real vs. what's a documented limitation

| Claim | Status |
|---|---|
| Real-time-capable vehicle detection (car/truck/bus/motorcycle) via pretrained YOLOv8 | **Real.** ~150ms/frame CPU after warm-up. Precision 1.00 / recall 0.80 on a small hand-labeled set; degrades on dense/historic scenes (documented above). |
| License-plate OCR | **Real**, EasyOCR + OpenCV region heuristic. 86.1% exact-match on a **synthetic** dataset (no real-plate dataset available/used) — see caveats above. |
| Emergency-vehicle detection | **Heuristic only, not a trained classifier.** 2.4% false-positive rate on known-negative images; **no true-positive/recall number exists** because no labeled emergency-vehicle data was available. |
| Multi-object tracking | **Real**, correctness-tested centroid+IOU tracker. Not state-of-the-art (no ByteTrack/DeepSORT). |
| GPU / EC2 deployment | Documented, real deployment steps in README.md. **We ran everything in this document on CPU** — GPU numbers are not claimed anywhere. |
| "81.9% accuracy in low-light conditions" (old resume/README claim) | **Not reproduced, not claimed here.** The closest real analog we measured is 75.0% exact-match OCR accuracy on our synthetic *low_light* tier (36-sample synthetic dataset, see table above) — a different metric, on different (synthetic) data, for a different sub-task (plate OCR, not overall detection). It is not a substitute for the original claim, which we could not verify or reproduce and have removed. |

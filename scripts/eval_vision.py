"""
Real evaluation script for the CV module. Run it directly:

    .venv\\Scripts\\python.exe scripts\\eval_vision.py

It does two independent things and prints (and optionally writes JSON for)
real, measured numbers — nothing here is hand-typed into results.md without
having actually come out of this script:

1. Vehicle detection accuracy on `tests/fixtures/*.jpg` against a small
   manually-annotated ground truth (see GROUND_TRUTH below — annotated by
   visually inspecting each image, documented per-image).

2. License-plate OCR accuracy on a synthetic plate dataset. We generate
   the plates ourselves (random alphanumeric strings rendered as text with
   varying blur/rotation/noise/brightness) because we do not have access
   to a permissively-licensed, privacy-safe dataset of real license
   plates. This is disclosed plainly in results.md — synthetic-plate
   accuracy is a proxy for OCR quality under controlled conditions, NOT a
   claim about real-world plate-reading accuracy.

Usage:
    python scripts/eval_vision.py --out results_raw.json
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"

# ── Part 1: vehicle detection ground truth ──────────────────────────────
# Manually annotated by visually inspecting each image. Images are only
# included here when vehicles could be counted with high confidence
# (i.e. clearly separable, not tiny/occluded background clutter). Very
# dense/historic scenes (auckland_traffic.jpg, herald_square_traffic.jpg)
# are intentionally EXCLUDED from precision/recall scoring below and
# reported separately as qualitative/raw-count-only, because manually
# hand-counting 40-90 tiny, overlapping vehicles is not reliable enough
# to treat as ground truth — see results.md for the raw counts anyway.
GROUND_TRUTH = {
    "bus.jpg": {
        "vehicles": 1,
        "classes": {"bus": 1},
        "note": "Ultralytics stock sample image; well-known composition (1 bus, several pedestrians).",
    },
    "zidane.jpg": {
        "vehicles": 0,
        "classes": {},
        "note": "Negative control — two soccer players, no vehicles anywhere in frame.",
    },
    "manhattan_50th_st.jpg": {
        "vehicles": 4,
        "classes": {"truck": 1, "bus": 1, "car": 2},
        "note": ("Foreground yellow box truck (International brand, headlights on), "
                 "a partially-occluded transit bus in the background, a yellow NYC taxi, "
                 "and a dark sedan/SUV behind the taxi."),
    },
}

QUALITATIVE_ONLY = ["auckland_traffic.jpg", "herald_square_traffic.jpg"]


def eval_detection():
    from app.ai.vision.pipeline import process_image

    per_image = []
    tp_total = fp_total = fn_total = 0

    for name, gt in GROUND_TRUTH.items():
        path = FIXTURES / name
        result = process_image(str(path), run_ocr=False, run_emergency=False)
        pred_classes = {}
        for v in result["vehicles"]:
            lbl = v["detection"]["label"]
            pred_classes[lbl] = pred_classes.get(lbl, 0) + 1

        pred_total = result["vehicle_count"]
        gt_total = gt["vehicles"]

        # Vehicle-level (class-agnostic) matching: did we find "a vehicle"
        # where one exists, regardless of whether the class label is exact.
        tp = min(pred_total, gt_total)
        fp = max(0, pred_total - gt_total)
        fn = max(0, gt_total - pred_total)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        # Class-exact accuracy: of predicted boxes, how many match the
        # expected class distribution (per-class min-count overlap).
        class_correct = sum(min(pred_classes.get(c, 0), n) for c, n in gt["classes"].items())

        per_image.append({
            "image": name,
            "ground_truth_vehicles": gt_total,
            "ground_truth_classes": gt["classes"],
            "predicted_vehicles": pred_total,
            "predicted_classes": pred_classes,
            "vehicle_level_tp": tp, "vehicle_level_fp": fp, "vehicle_level_fn": fn,
            "class_exact_correct": class_correct,
            "timing_ms": result["timing_ms"],
            "note": gt["note"],
        })

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else None
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else None

    qualitative = []
    for name in QUALITATIVE_ONLY:
        path = FIXTURES / name
        result = process_image(str(path), run_ocr=False, run_emergency=False)
        pred_classes = {}
        for v in result["vehicles"]:
            lbl = v["detection"]["label"]
            pred_classes[lbl] = pred_classes.get(lbl, 0) + 1
        qualitative.append({
            "image": name,
            "predicted_vehicles": result["vehicle_count"],
            "predicted_classes": pred_classes,
            "timing_ms": result["timing_ms"],
        })

    return {
        "per_image": per_image,
        "aggregate": {
            "vehicle_level_precision": precision,
            "vehicle_level_recall": recall,
            "tp": tp_total, "fp": fp_total, "fn": fn_total,
        },
        "qualitative_dense_scenes": qualitative,
    }


def eval_emergency_false_positive_rate():
    """
    All fixture images are known true negatives (no real emergency
    vehicle appears in any of them), so every `likely_emergency=True`
    flag on them is by definition a false positive. This gives a real,
    if narrow, measurement: the heuristic's false-positive rate on
    ordinary traffic scenes. We have no positive examples (no labeled
    emergency-vehicle images), so recall/precision on true emergency
    vehicles is NOT measured anywhere — that would require data we don't
    have, and we're not going to fake it.
    """
    from app.ai.vision.emergency import EMERGENCY_SCORE_THRESHOLD
    from app.ai.vision.pipeline import process_image

    all_images = list(GROUND_TRUTH.keys()) + QUALITATIVE_ONLY
    scores = []
    flagged = 0
    total = 0
    for name in all_images:
        result = process_image(str(FIXTURES / name), run_ocr=False, run_emergency=True)
        for v in result["vehicles"]:
            total += 1
            scores.append(v["emergency"]["emergency_score"])
            if v["emergency"]["likely_emergency"]:
                flagged += 1
    return {
        "threshold": EMERGENCY_SCORE_THRESHOLD,
        "total_vehicles_all_known_negative": total,
        "false_positives": flagged,
        "false_positive_rate": flagged / total if total else None,
        "score_distribution_summary": {
            "min": round(min(scores), 4) if scores else None,
            "median": round(sorted(scores)[len(scores) // 2], 4) if scores else None,
            "max": round(max(scores), 4) if scores else None,
        },
    }


# ── Part 2: synthetic plate OCR ─────────────────────────────────────────

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\cour.ttf",
]


def _make_plate_text(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=3))
    digits = "".join(rng.choices(string.digits, k=4))
    return f"{letters}{digits}"


def _render_plate(text: str, font_path: str, rng: random.Random, difficulty: str):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = 400, 120
    bg = (255, 255, 255) if difficulty != "low_light" else (60, 60, 62)
    fg = (10, 10, 10) if difficulty != "low_light" else (150, 150, 150)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 64)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2 - bbox[0], (h - th) / 2 - bbox[1]), text, font=font, fill=fg)

    # Thin border, like a real plate.
    draw.rectangle([2, 2, w - 3, h - 3], outline=(30, 30, 30), width=3)

    if difficulty in ("blur", "low_light"):
        img = img.rotate(rng.uniform(-6, 6), fillcolor=bg, expand=False)
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 2.2)))
    if difficulty == "low_light":
        arr = np.array(img).astype(np.float32)
        arr = arr * rng.uniform(0.55, 0.75)  # dim
        noise = np.random.default_rng(rng.randint(0, 10_000)).normal(0, 14, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        img = Image.fromarray(arr)

    return img


def build_synthetic_plate_dataset(n_per_tier: int = 12, seed: int = 42):
    rng = random.Random(seed)
    dataset = []
    for difficulty in ("clean", "blur", "low_light"):
        for _ in range(n_per_tier):
            text = _make_plate_text(rng)
            font_path = rng.choice(FONT_CANDIDATES)
            img = _render_plate(text, font_path, rng, difficulty)
            dataset.append({"text": text, "image": img, "difficulty": difficulty})
    return dataset


def eval_plate_ocr(n_per_tier: int = 12, seed: int = 42, save_dir: Path = None):
    import numpy as np

    from app.ai.vision.plate_ocr import get_plate_reader

    dataset = build_synthetic_plate_dataset(n_per_tier=n_per_tier, seed=seed)
    reader = get_plate_reader()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.perf_counter()
    for i, sample in enumerate(dataset):
        img_bgr = np.array(sample["image"])[:, :, ::-1].copy()  # RGB -> BGR
        h, w = img_bgr.shape[:2]
        read = reader.read_plate(img_bgr, (0, 0, w, h))
        pred_text = read.text if read else ""
        exact = pred_text == sample["text"]
        char_acc = _char_accuracy(sample["text"], pred_text)
        results.append({
            "index": i, "difficulty": sample["difficulty"], "ground_truth": sample["text"],
            "predicted": pred_text, "confidence": read.confidence if read else 0.0,
            "exact_match": exact, "char_accuracy": char_acc,
        })
        if save_dir:
            sample["image"].save(save_dir / f"{sample['difficulty']}_{i:02d}_{sample['text']}.png")
    total_time = time.perf_counter() - t0

    by_tier = {}
    for r in results:
        tier = by_tier.setdefault(r["difficulty"], {"n": 0, "exact": 0, "char_acc_sum": 0.0})
        tier["n"] += 1
        tier["exact"] += int(r["exact_match"])
        tier["char_acc_sum"] += r["char_accuracy"]

    summary = {}
    for tier, agg in by_tier.items():
        summary[tier] = {
            "n": agg["n"],
            "exact_match_accuracy": agg["exact"] / agg["n"],
            "avg_char_accuracy": agg["char_acc_sum"] / agg["n"],
        }

    overall_n = len(results)
    overall_exact = sum(r["exact_match"] for r in results)
    return {
        "per_sample": results,
        "by_difficulty": summary,
        "overall": {
            "n": overall_n,
            "exact_match_accuracy": overall_exact / overall_n,
            "avg_char_accuracy": sum(r["char_accuracy"] for r in results) / overall_n,
        },
        "total_time_s": round(total_time, 2),
        "avg_time_per_plate_ms": round(total_time / overall_n * 1000, 1),
    }


def _char_accuracy(gt: str, pred: str) -> float:
    if not gt:
        return 1.0 if not pred else 0.0
    matches = sum(1 for a, b in zip(gt, pred) if a == b)
    return matches / max(len(gt), len(pred)) if max(len(gt), len(pred)) else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="Write raw JSON results here")
    parser.add_argument("--save-plates", type=str, default=None,
                         help="Directory to save the generated synthetic plate images")
    parser.add_argument("--n-per-tier", type=int, default=12)
    args = parser.parse_args()

    print("=" * 70)
    print("VEHICLE DETECTION EVAL (real inference, real images)")
    print("=" * 70)
    det_results = eval_detection()
    for row in det_results["per_image"]:
        print(f"  {row['image']:30s} gt={row['ground_truth_vehicles']} "
              f"pred={row['predicted_vehicles']} classes_pred={row['predicted_classes']} "
              f"class_exact_correct={row['class_exact_correct']}/{row['ground_truth_vehicles']} "
              f"detect_ms={row['timing_ms']['detect']}")
    agg = det_results["aggregate"]
    print(f"\n  Vehicle-level precision={agg['vehicle_level_precision']:.3f} "
          f"recall={agg['vehicle_level_recall']:.3f} "
          f"(tp={agg['tp']} fp={agg['fp']} fn={agg['fn']})")
    print("\n  Qualitative (dense/historic scenes, no formal ground truth):")
    for row in det_results["qualitative_dense_scenes"]:
        print(f"    {row['image']:30s} predicted_vehicles={row['predicted_vehicles']} "
              f"classes={row['predicted_classes']}")

    print("\n" + "=" * 70)
    print("EMERGENCY-VEHICLE HEURISTIC: false-positive rate on known-negative images")
    print("=" * 70)
    em_results = eval_emergency_false_positive_rate()
    print(f"  threshold={em_results['threshold']} "
          f"total_vehicles={em_results['total_vehicles_all_known_negative']} "
          f"false_positives={em_results['false_positives']} "
          f"fp_rate={em_results['false_positive_rate']:.3f}")
    print(f"  score distribution: {em_results['score_distribution_summary']}")

    print("\n" + "=" * 70)
    print("LICENSE PLATE OCR EVAL (synthetic dataset, EasyOCR)")
    print("=" * 70)
    save_dir = Path(args.save_plates) if args.save_plates else None
    ocr_results = eval_plate_ocr(n_per_tier=args.n_per_tier, save_dir=save_dir)
    for tier, s in ocr_results["by_difficulty"].items():
        print(f"  {tier:12s} n={s['n']:3d}  exact_match_accuracy={s['exact_match_accuracy']:.3f}  "
              f"avg_char_accuracy={s['avg_char_accuracy']:.3f}")
    o = ocr_results["overall"]
    print(f"\n  Overall: n={o['n']} exact_match_accuracy={o['exact_match_accuracy']:.3f} "
          f"avg_char_accuracy={o['avg_char_accuracy']:.3f}")
    print(f"  Avg OCR time per plate: {ocr_results['avg_time_per_plate_ms']} ms")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"detection": det_results, "emergency_heuristic": em_results,
                        "plate_ocr": ocr_results}, f, indent=2)
        print(f"\nRaw results written to {args.out}")


if __name__ == "__main__":
    main()

"""Evaluate source-labeled photos without changing OncoEdge's production code.

Run from the repository root:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
    .venv/bin/python evaluation/evaluate_cases.py evaluation/2026-10-05/manifest.json

The baseline calls process_image unchanged. Whole-image classification, a lower
detection threshold, and correct BGR input are separate diagnostic experiments.
Published labels are references for this small convenience sample, not a clinical
gold standard. Detection counts do not establish correct localization.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.pipeline.inference_pipeline import OncoEdgePipeline


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxes_to_dicts(result):
    if result.boxes is None:
        return []
    return [
        {
            "bbox": [float(v) for v in box],
            "confidence": float(conf),
            "yolo_class": result.names[int(cls)],
        }
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        )
    ]


def classify_boxes(pipeline, image, boxes):
    """Reproduce the crop/fusion steps for a diagnostic detector experiment."""
    width, height = image.size
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if min(x2 - x1, y2 - y1) < 10:
            continue
        classification = pipeline.biomedclip.classify(image.crop((x1, y1, x2, y2)))
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "yolo_class": box["yolo_class"],
            "yolo_conf": box["confidence"],
            "class": classification["class"],
            "class_scores": classification["scores"],
            "fusion_score": pipeline.fusion.combine(
                box["confidence"], classification["max_score"]
            ),
        })
    return detections


def diagnostic_overlay(image, detections, path):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline="#ffd400", width=3)
        text = f'{detection["class"]} {detection["yolo_conf"]:.2f}'
        draw.text((x1 + 2, max(0, y1 - 12)), text, fill="#ffd400")
    annotated.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    baseline_revision = '1f195a099c4bdde1341375fc1368ace8f9408e9c'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if revision != baseline_revision or subprocess.call(
        ['git', 'diff', '--quiet', 'HEAD', '--', 'src'], cwd=ROOT
    ):
        raise SystemExit(
            'This script reproduces the historical, unchanged-code baseline. '
            'Use evaluation/evaluate_fixes.py for the locally fixed pipeline.'
        )
    manifest_path = args.manifest.resolve()
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    (directory / "baseline-overlays").mkdir(exist_ok=True)
    (directory / "bgr-diagnostic-overlays").mkdir(exist_ok=True)
    for case in manifest["cases"]:
        path = directory / case["file"]
        if sha256(path) != case["sha256"]:
            raise ValueError(f'Image hash mismatch: {case["id"]}')
        if case["visual_review_status"] != "reviewed":
            raise ValueError(f'Image not visually reviewed: {case["id"]}')

    torch.set_num_threads(2)
    started = time.perf_counter()
    pipeline = OncoEdgePipeline(device="cpu", force_backend="pytorch")
    initialization_seconds = time.perf_counter() - started
    original_detect = pipeline.yolo.detect
    observed = {}

    def observe_detect(*args, **kwargs):
        result = original_detect(*args, **kwargs)
        observed["result"] = result
        return result

    # This records the unchanged production detector result; it does not alter it.
    pipeline.yolo.detect = observe_detect
    metadata = manifest["patient_metadata"]
    high_factors = {"age": 65, "tobacco_years": 20, "lesion_duration": "> 6 weeks"}
    cases = []
    for case in manifest["cases"]:
        image = Image.open(directory / case["file"]).convert("RGB")
        array = np.array(image)
        started = time.perf_counter()
        output = pipeline.process_image(image, metadata)
        baseline_seconds = time.perf_counter() - started
        baseline_detections = [
            {key: value for key, value in detection.items() if key != "mask"}
            for detection in output["detections"]
        ]
        raw = boxes_to_dicts(observed["result"])
        Image.fromarray(output["visualization"]).save(
            directory / "baseline-overlays" / f'{case["id"]}.png'
        )

        # Isolation only: the production pipeline does NOT use this as a fallback.
        whole = pipeline.biomedclip.classify(image)
        # Ultralytics expects BGR arrays. Hold threshold/size/weights fixed.
        bgr = boxes_to_dicts(original_detect(np.ascontiguousarray(array[:, :, ::-1])))
        bgr_crops = classify_boxes(pipeline, image, bgr)
        diagnostic_overlay(image, bgr_crops,
                           directory / "bgr-diagnostic-overlays" / f'{case["id"]}.png')
        # Threshold diagnostic: same RGB behavior as production, only conf=0.10.
        threshold_boxes = boxes_to_dicts(original_detect(array, conf_threshold=0.10))
        max_fusion = max((d["fusion_score"] for d in baseline_detections), default=0)
        scenario = pipeline.decision_tree.assess_risk(max_fusion, **high_factors)
        bgr_fusion = max((d["fusion_score"] for d in bgr_crops), default=0)
        bgr_risk = pipeline.decision_tree.assess_risk(bgr_fusion, **metadata)
        row = {
            **case,
            "baseline": {
                "raw_yolo_boxes": raw,
                "detections": baseline_detections,
                "risk_level": output["risk_level"],
                "risk_score": output["risk_score"],
                "recommendation": output["recommendation"],
                "pipeline_seconds": baseline_seconds,
                "classifier_ran": bool(baseline_detections),
                "contains_reference_class": any(
                    d["class"] == case["expected_class"] for d in baseline_detections
                ),
            },
            "whole_image_classifier_diagnostic": whole,
            "bgr_input_diagnostic": {
                "raw_yolo_boxes": bgr,
                "detections": bgr_crops,
                "risk": bgr_risk,
            },
            "rgb_threshold_010_diagnostic": {"raw_yolo_boxes": threshold_boxes},
            "synthetic_high_factors_risk_only": scenario,
        }
        cases.append(row)
        print(json.dumps({
            "case": case["id"], "reference": case["expected_class"],
            "boxes": len(raw), "crop_classes": [d["class"] for d in baseline_detections],
            "risk": output["risk_level"], "score": round(output["risk_score"], 3),
            "whole_image": whole["class"], "bgr_boxes": len(bgr),
            "threshold_010_boxes": len(threshold_boxes),
        }), flush=True)
        # Save after every case so a interrupted run preserves completed evidence.
        (directory / "cases-results.json").write_text(json.dumps(cases, indent=2) + "\n")

    summary = {}
    for label in ["OCA", "OPMD", "Benign"]:
        group = [case for case in cases if case["expected_class"] == label]
        summary[label] = {
            "cases": len(group),
            "images_with_baseline_detection": sum(bool(c["baseline"]["detections"]) for c in group),
            "images_with_any_reference_class_crop": sum(c["baseline"]["contains_reference_class"] for c in group),
            "whole_image_reference_class_agreement": sum(
                c["whole_image_classifier_diagnostic"]["class"] == label for c in group
            ),
            "images_with_bgr_detection": sum(bool(c["bgr_input_diagnostic"]["detections"]) for c in group),
            "baseline_risk_levels": dict(Counter(c["baseline"]["risk_level"] for c in group)),
        }
    result = {
        "date": "2026-10-05", "manifest_sha256": sha256(manifest_path),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "backend": "pytorch", "device": "cpu", "host": platform.platform(),
        "python": platform.python_version(), "torch_threads": torch.get_num_threads(),
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "ultralytics", "open_clip_torch", "Pillow"]},
        "yolo_checkpoint_sha256": sha256(ROOT / "models/best.pt"),
        "biomedclip_checkpoint_sha256": sha256(ROOT / "models/biomedclip/open_clip_pytorch_model.bin"),
        "detector_threshold": 0.25, "detector_image_size": 640,
        "classifier_prompts": pipeline.biomedclip.PROMPTS,
        "patient_metadata": metadata, "synthetic_high_factors": high_factors,
        "initialization_seconds": initialization_seconds,
        "summary": summary,
        "limitations": [
            "Selected internet cases, not a representative or independent clinical test set.",
            "Source captions/case descriptions are reference labels, not verified ground truth.",
            "No bounding-box ground truth; presence of a detection does not prove correct localization.",
            "No healthy controls; specificity for healthy mucosa cannot be measured.",
            "Patient identities and cross-case independence are unknown in review articles.",
            "Training photo binaries unavailable; overlap with YOLO training cannot be ruled out.",
            "Pretraining overlap is possible, including published images in BioMedCLIP's training corpus.",
            "Most PDF photos are compressed/low resolution; source detail was not enhanced.",
            "Whole-image softmax scores are relative prompt scores, not calibrated disease probabilities.",
            "Timings are local CPU measurements, not hosted AWS latency or Jetson benchmarks.",
        ],
        "cases": cases,
    }
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

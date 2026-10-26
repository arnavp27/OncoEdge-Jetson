"""Compare plausible inference fixes without changing the app or saved baselines.

This is an exploratory replay of an already inspected convenience sample, not
an independent test. Source labels describe entire photographs; crop agreement
is only a diagnostic proxy, not classification accuracy or localization recall.

Run offline from the repository root using the existing uv environment:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
    .venv/bin/python evaluation/diagnose_abstention.py
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageOps
import torch

from src.config import CONFIG
from src.models.biomedclip_classifier import BiomedCLIPClassifier
from src.models.yolo_detector import YOLODetector


CLASS_NAMES = BiomedCLIPClassifier.CLASS_NAMES
BROAD_LABELS = (
    "oral squamous cell carcinoma", "oral potentially malignant disorder",
    "benign oral lesion", "normal oral mucosa",
)
TEMPLATES = ("clinical photograph of {}", "this is a photo of {}")


def encode_prompts(classifier, prompts):
    tokens = torch.cat([
        classifier.tokenizer.tokenizer(
            prompt, max_length=256, padding="max_length", truncation=True,
            return_tensors="pt",
        )["input_ids"] for prompt in prompts
    ])
    with torch.inference_mode():
        features = classifier.model.encode_text(tokens)
        return torch.nn.functional.normalize(features, dim=-1)


def evaluate_region(feature, text, scale, region):
    scores = (scale * feature @ text.T).softmax(-1).flatten().tolist()
    index = int(np.argmax(scores))
    label = CLASS_NAMES[index]
    ranked = sorted(scores, reverse=True)
    margin = ranked[0] - ranked[1]
    disagreement = region["yolo_class"] != label and not (
        region["yolo_class"] == "Benign" and label == "Normal"
    )
    uncertain = (
        ranked[0] < CONFIG["model"].min_classifier_score
        or margin < CONFIG["model"].min_classifier_margin
    )
    return {
        **region, "class": label, "scores": dict(zip(CLASS_NAMES, scores)),
        "margin": margin, "disagreement": disagreement,
        "uncertain": uncertain, "resolved": not (disagreement or uncertain),
    }


def summarize(cases):
    complete = [c for c in cases if c["regions"] and all(
        region["resolved"] for region in c["regions"]
    )]
    # A photograph can contain several regions with different true labels.
    # These counts must therefore never be described as crop accuracy.
    return {
        "photos": len(cases),
        "photos_with_regions": sum(bool(c["regions"]) for c in cases),
        "all_regions_resolved_photos": len(complete),
        "all_regions_resolved_ids": [c["id"] for c in complete],
        "no_region_ids": [c["id"] for c in cases if not c["regions"]],
        "photos_with_any_reference_matching_crop": sum(any(
            r["class"] == c["reference"] for r in c["regions"]
        ) for c in cases),
        "resolved_crop_label_counts": dict(Counter(
            r["class"] for c in cases for r in c["regions"] if r["resolved"]
        )),
        "resolved_crop_source_label_mismatches": [
            {"id": c["id"], "reference": c["reference"],
             "region": i + 1, "class": r["class"]}
            for c in cases for i, r in enumerate(c["regions"])
            if r["resolved"] and r["class"] != c["reference"]
        ],
        "region_disagreements": sum(r["disagreement"] for c in cases for r in c["regions"]),
        "region_ambiguous_scores": sum(r["uncertain"] for c in cases for r in c["regions"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evaluation/2026-10-23/abstention-experiment.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists. Choose a new --output to preserve earlier results.")
    directory = ROOT / "evaluation/2026-10-05"
    source_paths = [directory / "manifest.json", directory / "fixed-results.json"]
    saved_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source_paths}
    manifest = json.loads(source_paths[0].read_text())
    saved = {c["id"]: c for c in json.loads(source_paths[1].read_text())["cases"]}
    torch.set_num_threads(2)
    classifier = BiomedCLIPClassifier(device="cpu")
    detector = YOLODetector(device="cpu")
    learned_scale = float(classifier.model.logit_scale.exp().item())
    broad = encode_prompts(classifier, [TEMPLATES[0].format(s) for s in BROAD_LABELS])
    ensemble = encode_prompts(classifier, [
        template.format(label) for label in BROAD_LABELS for template in TEMPLATES
    ]).reshape(4, len(TEMPLATES), -1).mean(1)
    ensemble = torch.nn.functional.normalize(ensemble, dim=-1)
    methods = {
        "deployed_baseline": ("tight", classifier.text_features, 100.0),
        "checkpoint_temperature": ("tight", classifier.text_features, learned_scale),
        "preserve_entire_crop": ("padded", classifier.text_features, 100.0),
        "add_crop_context": ("context", classifier.text_features, 100.0),
        "broad_class_prompts": ("tight", broad, 100.0),
        "broad_prompt_ensemble": ("tight", ensemble, 100.0),
    }
    reports = {name: [] for name in methods}
    detector_512 = []
    preprocessing = str(classifier.preprocess)
    for case in manifest["cases"]:
        path = directory / case["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
        image = Image.open(path).convert("RGB")
        rows = {name: [] for name in methods}
        for region in saved[case["id"]]["detections"]:
            x1, y1, x2, y2 = region["bbox"]
            crop = image.crop((x1, y1, x2, y2))
            # Padding retains the full crop instead of clipping the long axis.
            side = max(crop.size)
            padded = ImageOps.pad(crop, (side, side), color=(123, 117, 104))
            dx, dy = (x2 - x1) * .2, (y2 - y1) * .2
            context = image.crop((max(0, int(x1 - dx)), max(0, int(y1 - dy)),
                                  min(image.width, int(x2 + dx)),
                                  min(image.height, int(y2 + dy))))
            tensors = torch.stack([classifier.preprocess(c) for c in (crop, padded, context)])
            with torch.inference_mode():
                features = classifier.model.encode_image(tensors)
                features = torch.nn.functional.normalize(features, dim=-1)
            by_crop = dict(zip(("tight", "padded", "context"), features))
            base = {k: region[k] for k in ("bbox", "yolo_class", "yolo_conf")}
            for name, (crop_name, text, scale) in methods.items():
                result = evaluate_region(by_crop[crop_name], text, scale, base)
                rows[name].append(result)
                if name == "deployed_baseline":
                    assert result["class"] == region["class"], case["id"]
                    assert max(abs(result["scores"][k] - v)
                               for k, v in region["class_scores"].items()) < 1e-4
        for name in methods:
            reports[name].append({"id": case["id"], "reference": case["expected_class"],
                                  "regions": rows[name]})
        # Independent detector hypothesis: train-time image size, same .25 gate.
        detections = detector.model.predict(
            source=np.asarray(image)[:, :, ::-1].copy(), device="cpu",
            imgsz=512, conf=CONFIG["model"].yolo_conf_threshold, verbose=False,
        )[0]
        detector_512.append({"id": case["id"], "reference": case["expected_class"],
                             "regions": len(detections.boxes),
                             "labels": [detections.names[int(i)] for i in detections.boxes.cls],
                             "confidence": detections.boxes.conf.tolist()})
        print(case["id"], "baseline_regions", len(rows["deployed_baseline"]),
              "imgsz512_regions", len(detections.boxes), flush=True)
    results = {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=ROOT, text=True).strip(),
        "source_hashes": saved_hashes,
        "experiment_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "models/best.pt", ROOT / "models/biomedclip/open_clip_pytorch_model.bin")
        },
        "checkpoint_logit_scale": learned_scale, "preprocess": preprocessing,
        "class_names": CLASS_NAMES, "broad_labels": BROAD_LABELS,
        "prompt_templates": TEMPLATES,
        "gates": {"min_score": CONFIG["model"].min_classifier_score,
                  "min_margin": CONFIG["model"].min_classifier_margin},
        "variants": {name: {"summary": summarize(cases), "cases": cases}
                     for name, cases in reports.items()},
        "detector_imgsz512": {
            "photos_with_regions": sum(c["regions"] > 0 for c in detector_512),
            "cases": detector_512,
        },
        "limitations": [
            "17 previously inspected publication photographs; development diagnostics only.",
            "Photo source labels are not ground truth for every detector crop.",
            "No localization annotations, independent test set, normal controls or calibration.",
            "No changes to application code, thresholds, deployed models or AWS resources.",
            "No inference variant is selected for production from this small replay.",
        ],
    }
    for path in source_paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved_hashes[str(path.relative_to(ROOT))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for name, result in results["variants"].items():
        print(name, json.dumps(result["summary"]), flush=True)
    print("REPORT", args.output, flush=True)


if __name__ == "__main__":
    main()

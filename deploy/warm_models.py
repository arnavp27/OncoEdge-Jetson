"""Load the real CPU pipeline before exposing the demo to visitors."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.inference_pipeline import OncoEdgePipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Also run both sample images and the classifier.")
    args = parser.parse_args()
    if torch.version.cuda is not None:
        raise SystemExit("The hosted demo must use the CPU-only PyTorch build.")

    started = time.monotonic()
    pipeline = OncoEdgePipeline(device="cpu", force_backend="pytorch")
    expected_classes = {0: "OCA", 1: "OPMD", 2: "Benign"}
    if pipeline.yolo.model.names != expected_classes:
        raise SystemExit(f"Unexpected detector classes: {pipeline.yolo.model.names}")
    if torch.any(torch.pdist(pipeline.biomedclip.text_features.float()) <= 1e-6):
        raise SystemExit("Clinical prompts produced identical embeddings; check the tokenizer vocabulary.")
    print(f"Models ready in {time.monotonic() - started:.1f}s; detector classes: {expected_classes}")

    if args.smoke_test:
        metadata = {"age": 45, "tobacco_years": 0, "lesion_duration": "< 2 weeks"}
        for filename in ("027.jpeg", "photo.webp"):
            with Image.open(PROJECT_ROOT / filename) as sample:
                image = sample.convert("RGB")
            started = time.monotonic()
            result = pipeline.process_image(np.array(image), metadata)
            # Exercise the classifier even if this sample has no detector candidates.
            classification = pipeline.biomedclip.classify(image)
            scores = classification["scores"]
            if set(scores) != {"OCA", "OPMD", "Benign", "Normal"}:
                raise SystemExit(f"Unexpected classifier labels: {scores}")
            if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores.values()):
                raise SystemExit(f"Invalid classifier scores: {scores}")
            if not np.isclose(sum(scores.values()), 1.0, atol=1e-5):
                raise SystemExit(f"Classifier scores do not sum to one: {scores}")
            print(json.dumps({
                "sample": filename,
                "seconds": round(time.monotonic() - started, 2),
                "detections": len(result["detections"]),
                "risk_level": result["risk_level"],
                "risk_score": result["risk_score"],
                "classifier": classification,
            }))

    memory = {"process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    container_peak = Path("/sys/fs/cgroup/memory.peak")
    if Path("/.dockerenv").is_file() and container_peak.is_file():
        memory["container_peak_memory_mib"] = round(int(container_peak.read_text()) / 1024**2, 1)
    print(json.dumps(memory))


if __name__ == "__main__":
    main()

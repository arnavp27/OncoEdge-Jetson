"""
OncoEdge — YOLO11s-Seg Fine-Tuning for Oral Cancer Segmentation
================================================================
Training + Validation ONLY. Export is handled by export_model.py.

Classes:
    0: OCA  — Oral Cancer (Squamous Cell Carcinoma)
    1: OPMD — Oral Potentially Malignant Disorder
    2: Benign — Ulcers, VBD, Candidiasis, etc.

Negative Samples:
    Healthy images with empty .txt labels are included automatically.
    YOLO treats empty label files as background-only (no objects).

Usage:
    python train_oral_cancer.py

For Google Colab, change the 3 paths in the CONFIGURATION section below.
"""

import subprocess
import sys
import os
from pathlib import Path

# =====================================================================
# 0. DEPENDENCY CHECK
# =====================================================================
def ensure_ultralytics():
    """Install ultralytics if not already present."""
    try:
        import ultralytics
        print(f"[✓] ultralytics {ultralytics.__version__} already installed")
    except ImportError:
        print("[!] Installing ultralytics...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "ultralytics", "--quiet"
        ])
        import ultralytics
        print(f"[✓] ultralytics {ultralytics.__version__} installed")

ensure_ultralytics()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURATION — Change these 3 paths for Colab portability
# =====================================================================
# --- Local Windows paths ---
DATA_YAML   = r"E:\4thYear\EMbedding_AI\Project\datasets\oral_cancer_seg\data.yaml"
PROJECT_DIR = r"E:\4thYear\EMbedding_AI\Project\zenodo_DOAOC\yolo_finetune"
BASE_MODEL  = "yolo11s-seg.pt"  # Downloaded automatically by ultralytics

# --- Uncomment below for Google Colab ---
# DATA_YAML   = "/content/datasets/oral_cancer_seg/data.yaml"
# PROJECT_DIR = "/content/yolo_finetune"
# BASE_MODEL  = "yolo11s-seg.pt"

# =====================================================================
# 2. HYPERPARAMETERS — Tuned for Medical Oral Cavity Imaging
# =====================================================================
HYPERPARAMS = dict(
    # ── Core Training ──────────────────────────────────────────────
    epochs      = 100,
    imgsz       = 512,           # Reduced from 640 for 4GB VRAM
    batch       = 4,             # Reduced from 16 for 4GB VRAM
    device      = 0,             # GPU 0; use 'cpu' if no GPU

    # ── Early Stopping ─────────────────────────────────────────────
    patience    = 15,           # Stop if val loss plateaus for 15 epochs

    # ── Learning Rate ──────────────────────────────────────────────
    lr0         = 0.01,         # Initial LR (SGD default)
    lrf         = 0.01,         # Final LR = lr0 * lrf (cosine decay)
    warmup_epochs = 3.0,        # Gradual warmup

    # ── Optimizer ──────────────────────────────────────────────────
    optimizer   = "SGD",        # SGD with momentum; alt: "AdamW"
    momentum    = 0.937,
    weight_decay = 0.0005,

    # ── Augmentations (Medical-Specific) ───────────────────────────
    # Oral images have high lighting variance from dental lamps
    hsv_h       = 0.015,        # Hue jitter (subtle — lesion color matters)
    hsv_s       = 0.7,          # Saturation jitter (compensate flash)
    hsv_v       = 0.4,          # Value/brightness jitter (shadow/glare)
    degrees     = 10.0,         # Slight rotation (mouth angle varies)
    translate   = 0.1,          # Small translation
    scale       = 0.5,          # Scale augmentation
    flipud      = 0.5,          # Vertical flip (valid for oral cavity)
    fliplr      = 0.5,          # Horizontal flip
    mosaic      = 0.0,          # Disabled — uses 4x memory (4GB VRAM limit)
    mixup       = 0.0,          # Disabled — uses 2x memory
    copy_paste  = 0.0,          # Disabled — high VRAM cost

    # ── Loss Weights ───────────────────────────────────────────────
    box         = 7.5,          # Box loss weight
    cls         = 0.5,          # Classification loss weight
    dfl         = 1.5,          # Distribution focal loss weight

    # ── Output ─────────────────────────────────────────────────────
    project     = PROJECT_DIR,
    name        = "oncoedge_v1",
    exist_ok    = True,         # Overwrite previous run
    save        = True,
    save_period  = 10,          # Checkpoint every 10 epochs
    plots       = True,         # Generate training plots
    verbose     = True,
)


# =====================================================================
# 3. TRAINING
# =====================================================================
def train():
    print("=" * 60)
    print("  OncoEdge — YOLO11s-Seg Training")
    print("  Model:   ", BASE_MODEL)
    print("  Dataset: ", DATA_YAML)
    print("  Output:  ", os.path.join(PROJECT_DIR, "oncoedge_v1"))
    print("=" * 60)

    # Load pre-trained model
    model = YOLO(BASE_MODEL)

    # Verify data.yaml exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"data.yaml not found at: {DATA_YAML}\n"
            f"Run convert_coco_to_yolo_seg.py first."
        )

    # Launch training
    results = model.train(
        data=DATA_YAML,
        **HYPERPARAMS
    )

    return model, results


# =====================================================================
# 4. VALIDATION — Print Box + Mask mAP metrics
# =====================================================================
def validate(model):
    print("\n" + "=" * 60)
    print("  VALIDATION — Best Weights")
    print("=" * 60)

    # Load best weights from training run
    best_weights = os.path.join(
        PROJECT_DIR, "oncoedge_v1", "weights", "best.pt"
    )

    if os.path.exists(best_weights):
        val_model = YOLO(best_weights)
    else:
        print("[!] best.pt not found, using last trained model")
        val_model = model

    metrics = val_model.val(
        data=DATA_YAML,
        imgsz=512,
        split="val",
        plots=True,
        save_json=True,
    )

    # ── Print Results ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  BOX DETECTION METRICS")
    print("─" * 50)
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    print("\n" + "─" * 50)
    print("  MASK SEGMENTATION METRICS")
    print("─" * 50)
    print(f"  mAP50      : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95   : {metrics.seg.map:.4f}")
    print(f"  Precision   : {metrics.seg.mp:.4f}")
    print(f"  Recall      : {metrics.seg.mr:.4f}")

    # Per-class breakdown
    class_names = ["OCA", "OPMD", "Benign"]
    print("\n" + "─" * 50)
    print("  PER-CLASS mAP50 (Box | Mask)")
    print("─" * 50)
    for i, name in enumerate(class_names):
        try:
            box_ap = metrics.box.maps[i]
            seg_ap = metrics.seg.maps[i]
            print(f"  {name:8s}  →  Box: {box_ap:.4f}  |  Mask: {seg_ap:.4f}")
        except (IndexError, AttributeError):
            print(f"  {name:8s}  →  (no predictions for this class)")

    return metrics


# =====================================================================
# 5. MAIN — Train + Validate (NO Export)
# =====================================================================
if __name__ == "__main__":
    # Train
    model, results = train()

    # Validate & print metrics
    metrics = validate(model)

    # ── Final Summary ──────────────────────────────────────────────
    best_path = os.path.join(PROJECT_DIR, "oncoedge_v1", "weights", "best.pt")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best weights saved at: {best_path}")
    print(f"  Training plots:        {os.path.join(PROJECT_DIR, 'oncoedge_v1')}")
    print(f"\n  To resume training:")
    print(f"    yolo segment train model={PROJECT_DIR}\\oncoedge_v1\\weights\\last.pt resume=True")
    print(f"\n  To export to ONNX:")
    print(f"    python export_model.py")

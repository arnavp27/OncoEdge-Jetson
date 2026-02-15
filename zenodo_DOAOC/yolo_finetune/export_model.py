"""
OncoEdge — ONNX Export for Jetson Nano / Orin Deployment
========================================================
Exports the trained best.pt to ONNX format.
Run AFTER training is complete (train_oral_cancer.py).

Usage:
    python export_model.py

For Google Colab, change WEIGHTS_PATH below.
"""

import subprocess
import sys
import os

# =====================================================================
# 0. DEPENDENCY CHECK
# =====================================================================
def ensure_ultralytics():
    try:
        import ultralytics
        print(f"[✓] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[!] Installing ultralytics...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "ultralytics", "--quiet"
        ])

ensure_ultralytics()

from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
# --- Local Windows ---
WEIGHTS_DIR = r"E:\4thYear\EMbedding_AI\Project\zenodo_DOAOC\yolo_finetune\oncoedge_v1\weights"

# --- Uncomment for Google Colab ---
# WEIGHTS_DIR = "/content/yolo_finetune/oncoedge_v1/weights"

BEST_PT = os.path.join(WEIGHTS_DIR, "best.pt")

# =====================================================================
# 2. EXPORT TO ONNX
# =====================================================================
def export():
    print("=" * 60)
    print("  OncoEdge — ONNX Export")
    print("=" * 60)

    # Verify best.pt exists
    if not os.path.exists(BEST_PT):
        raise FileNotFoundError(
            f"best.pt not found at: {BEST_PT}\n"
            f"Run train_oral_cancer.py first to generate trained weights."
        )

    print(f"  Loading: {BEST_PT}")
    model = YOLO(BEST_PT)

    # Export
    onnx_path = model.export(
        format="onnx",
        opset=16,               # Jetson Nano / Orin compatible
        imgsz=512,              # Must match training imgsz
        simplify=True,          # ONNX simplifier for edge deployment
        dynamic=False,          # Fixed shape for TensorRT conversion
        half=False,             # FP32; convert to FP16 on device
    )

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXPORT COMPLETE")
    print("=" * 60)
    print(f"  ONNX model saved: {onnx_path}")
    print(f"\n  For Jetson Nano/Orin, convert to TensorRT:")
    print(f"    /usr/src/tensorrt/bin/trtexec \\")
    print(f"      --onnx={onnx_path} \\")
    print(f"      --saveEngine=oncoedge_v1.engine \\")
    print(f"      --fp16")
    print(f"\n  Or use ultralytics directly on Jetson:")
    print(f"    yolo predict model={onnx_path} source=image.jpg")

    return onnx_path


# =====================================================================
# 3. MAIN
# =====================================================================
if __name__ == "__main__":
    export()

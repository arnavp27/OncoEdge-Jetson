"""
YOLOv11 Export Script for Jetson Nano
=====================================
Exports the fine-tuned model to ONNX with specific settings
guaranteed to work on Jetson Nano (JetPack 4.6).
"""
from ultralytics import YOLO
import os

# CONFIGURATION
# 1. Path to your trained weights
MODEL_PATH = r"E:\4thYear\EMbedding_AI\Project\zenodo_DOAOC\yolo_finetune\oncoedge_v1\weights\best.pt"
# 2. Path to save the converted file
OUTPUT_DIR = r"E:\4thYear\EMbedding_AI\Project\zenodo_DOAOC\yolo_finetune\converted"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print("\nStarting Export for Jetson Nano...")
    print("Settings:")
    print("  - Format: ONNX")
    print("  - Opset:  11 (REQUIRED for JetPack 4.6 compatibility)")
    print("  - Size:   512x512 (Matching training)")
    print("  - Batch:  1 (Fixed batch size is faster on Nano)")
    
    # EXPORT
    output_path = model.export(
        format="onnx",
        imgsz=512,       # Match training size
        dynamic=False,   # STATIC input shape is best for Nano
        opset=11,        # CRITICAL: Opset 16 will fail on Nano
        simplify=True,   # Clean up the graph
        batch=1          # Optimized for single-image inference
    )
    
    # Move and Rename
    generated_file = MODEL_PATH.replace(".pt", ".onnx")
    final_file = os.path.join(OUTPUT_DIR, "yolo_oncoedge_nano.onnx")
    
    if os.path.exists(generated_file):
        if os.path.exists(final_file):
            os.remove(final_file)
        os.rename(generated_file, final_file)
        print(f"\n[SUCCESS] Exported to: {final_file}")
        print("\nNEXT STEP (On Jetson Nano):")
        print(f"Run this command to build the fast engine:")
        print(f"/usr/src/tensorrt/bin/trtexec --onnx=yolo_oncoedge_nano.onnx --saveEngine=yolo.engine --fp16")
    else:
        print("\n[ERROR] Export failed. Check logs.")

if __name__ == "__main__":
    main()
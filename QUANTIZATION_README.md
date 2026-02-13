# BiomedCLIP Quantization for Jetson Nano - Progress Report

## ✅ Completed Steps (Windows Development)

### Phase 0: Shared Preprocessing Module
**File**: `src/models/biomedclip_preprocess.py`
- ✅ Created shared preprocessing module
- ✅ Frozen OpenCLIP parameters (BICUBIC interpolation, ImageNet normalization)
- ✅ Single source of truth for both calibration and inference
- ✅ Prevents circular imports and preprocessing drift

### Phase 1: Calibration Data Preparation
**File**: `src/utils/prepare_calibration_data.py`
- ✅ Created calibration data collection script
- ⏳ **Pending**: Collect 250-400 raw JPEG images
- **Usage**: `python src/utils/prepare_calibration_data.py --source-dirs <image_dirs> --count 300`

### Phase 2: Text Embeddings Precomputation
**File**: `models/biomedclip/text_embeddings.npy`
- ✅ Precomputed text embeddings for 3 clinical prompts
- ✅ Size: 6.12 KB (3 × 512 features)
- ✅ Eliminates need to deploy 440MB text encoder!
- **Classes**: OSCC, OPMD, Normal

### Phase 3: CLIP Logit Scale Extraction
**File**: `models/biomedclip/logit_scale.json`
- ✅ Extracted learned temperature parameter
- ✅ Value: **85.23** (within expected range ~100.0)
- ✅ Ensures confidence calibration after INT8 quantization

### Phase 4: Vision Encoder ONNX Export
**Files**:
- `models/biomedclip/onnx/vision_encoder.onnx` (824 KB)
- `models/biomedclip/onnx/vision_encoder.onnx.data` (329 MB)
- **Total**: ~330 MB FP32
- ✅ Opset 13 for Jetson Nano compatibility
- ✅ Fixed batch size (1, 3, 224, 224) → (1, 512) features
- ✅ Validated with ONNX checker

### Phase 5: ONNX Validation Testing
**File**: `tests/test_onnx_export.py`
- ✅ All 3 tests passing:
  - ✅ ONNX model file exists
  - ✅ ONNX output matches PyTorch output (rtol=1e-3, atol=1e-5)
  - ✅ I/O shapes correct: (1,3,224,224) → (1,512)

---

## 📦 Files Ready for Jetson Nano Transfer

When you get your Jetson Nano, transfer these files:

```
oncoedge/
├── models/
│   └── biomedclip/
│       ├── onnx/
│       │   ├── vision_encoder.onnx          # 824 KB
│       │   └── vision_encoder.onnx.data     # 329 MB
│       ├── text_embeddings.npy              # 6 KB (3×512)
│       └── logit_scale.json                 # <1 KB (temp=85.23)
├── data/
│   └── calibration_images/                  # 250-400 JPEGs (when collected)
└── src/
    └── models/
        └── biomedclip_preprocess.py         # Shared preprocessing
```

---

## 🔧 Next Steps

### On Windows (Pending):
1. **Collect calibration images** (250-400 diverse oral cavity images)
   ```bash
   python src/utils/prepare_calibration_data.py \
     --source-dirs path/to/images1 path/to/images2 \
     --count 300
   ```

### On Jetson Nano (Future):
2. **Build INT8 TensorRT engine** (15-30 min build time)
   ```bash
   python3 src/models/build_tensorrt_engine.py
   ```

3. **Create TensorRT inference wrapper**
   - File: `src/models/biomedclip_tensorrt.py`
   - Loads: INT8 engine + text embeddings + logit scale

4. **Integrate with OncoEdge pipeline**
   - Auto-detect Jetson Nano vs Windows
   - Fallback to PyTorch if TensorRT not available

---

## 📊 Expected Performance on Jetson Nano

| Metric | PyTorch FP32 | TensorRT INT8 | Improvement |
|--------|--------------|---------------|-------------|
| Model Size | 344 MB | ~70-120 MB | 3-5× smaller |
| Memory Usage | ~1.8-2.2 GB | ~800MB-1.3GB | ~1.5-2× less |
| Vision Latency | 400-600 ms | 200-350 ms | 1.5-2× faster |
| Total Pipeline | N/A | 335-610 ms | **1.6-3 FPS** |

**Note**: FP16 engine is recommended to try first (simpler, faster build, similar performance).

---

## 🔍 Architecture Details

### BiomedCLIP Model:
- **Vision Encoder**: ViT-B/16 (86M params)
  - Input: (1, 3, 224, 224) RGB image
  - Output: (1, 512) feature vector
  - **Quantization target**: INT8 TensorRT on Jetson Nano

- **Text Encoder**: PubMedBERT (110M params)
  - Input: 3 clinical prompts (tokenized)
  - Output: (3, 512) feature vectors
  - **Optimization**: Precomputed on Windows (eliminates 440MB!)

### Similarity Computation:
```python
image_features = vision_encoder(image)           # (1, 512)
image_features = normalize(image_features)       # L2 norm

text_features = precomputed_embeddings           # (3, 512) - already normalized

similarity = image_features @ text_features.T    # (1, 3)
scaled = similarity * logit_scale                # Temperature scaling (85.23)
scores = softmax(scaled)                         # Final probabilities
```

---

## 🛠️ Dependencies Added

Updated `requirements.txt` with:
```
onnx==1.20.1
onnxscript==0.6.2
onnxruntime==1.21.1  # For validation only
```

---

## ✅ Validation Results

**ONNX Export Accuracy**:
- Output matches PyTorch within numerical precision (rtol=1e-3)
- All I/O shapes correct
- Model structure validated with ONNX checker

**Ready for Jetson Nano deployment!**

---

## 📚 References

- Full quantization plan: `C:\Users\1aapa\.claude\plans\abstract-pondering-wand.md`
- Original plan file: `plans/biomedclip_quantization_plan.md`
- Production readiness: **9.2/10** (ChatGPT assessment)
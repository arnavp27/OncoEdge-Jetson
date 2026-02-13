# Jetson Nano TensorRT Deployment Guide

Complete step-by-step instructions for deploying BiomedCLIP INT8 TensorRT engine on Jetson Nano.

---

## Prerequisites Checklist

### Files Already Transferred to Jetson ✓

Verify these files exist on your Jetson Nano:

```bash
# Check ONNX model
ls -lh models/biomedclip/onnx/vision_encoder.onnx
ls -lh models/biomedclip/onnx/vision_encoder.onnx.data

# Check precomputed data
ls -lh models/biomedclip/text_embeddings.npy
ls -lh models/biomedclip/logit_scale.json

# Check calibration images (should show 250-400)
ls data/calibration_images/*.jpg | wc -l

# Check preprocessing module
ls src/models/biomedclip_preprocess.py
```

**Expected Output:**
- `vision_encoder.onnx`: ~824 KB
- `vision_encoder.onnx.data`: ~329 MB
- `text_embeddings.npy`: 6.12 KB
- `logit_scale.json`: <1 KB
- Calibration images: 250-400 JPEGs

### Jetson System Requirements

```bash
# Check Jetson model
cat /etc/nv_tegra_release
# Expected: NVIDIA Jetson Nano

# Check JetPack version
dpkg -l | grep nvidia-jetpack
# Expected: JetPack 4.6.x (includes TensorRT)

# Check CUDA
nvcc --version
# Expected: CUDA 10.2

# Check Python 3
python3 --version
# Expected: Python 3.6+

# Check available disk space (need >500MB)
df -h .
```

### Python Dependencies

```bash
# Install/verify required packages
pip3 install pycuda numpy pillow

# Check TensorRT (should already be installed with JetPack)
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"

# Verify CUDA works
python3 -c "import pycuda.autoinit; print('CUDA initialized successfully')"
```

---

## Step 1: Build INT8 TensorRT Engine

**Estimated time:** 15-30 minutes

### Command

```bash
# Navigate to project root
cd /path/to/oncoedge

# Build INT8 engine with calibration
python3 src/models/build_tensorrt_engine.py \
    --onnx models/biomedclip/onnx/vision_encoder.onnx \
    --calib-dir data/calibration_images \
    --output models/biomedclip/tensorrt/vision_encoder_int8.engine \
    --workspace 512
```

### What to Expect

The build process will:
1. Load FP32 ONNX model (~330MB)
2. Parse ONNX network
3. Run INT8 calibration on 250-400 images (this is the slow part)
4. Optimize and build engine
5. Save engine (~70-120MB) and calibration cache

**Console Output:**
```
======================================================================
BiomedCLIP TensorRT INT8 Engine Builder
======================================================================
[OK] Found ONNX model with external data:
     models/biomedclip/onnx/vision_encoder.onnx (824.00 KB)
     models/biomedclip/onnx/vision_encoder.onnx.data (329.00 MB)

[1/5] Creating TensorRT builder...
[2/5] Parsing ONNX model: models/biomedclip/onnx/vision_encoder.onnx
[OK] ONNX model parsed successfully
     Network inputs: ['image']
     Network outputs: ['image_features']
[3/5] Configuring builder (workspace: 512MB)...
[OK] INT8 quantization enabled (with FP16 fallback for unstable layers)
[4/5] Initializing INT8 calibrator...
[OK] Calibrator initialized with 300 images

[OK] Starting TensorRT engine build...
     This will take 15-30 minutes on Jetson Nano
     Calibration will run on Nano GPU with 300 images
     Note: JPEG decoding is CPU-bound during build (expected)

[5/5] Building INT8 engine (please wait)...
  Calibration progress: 50/300
  Calibration progress: 100/300
  Calibration progress: 150/300
  Calibration progress: 200/300
  Calibration progress: 250/300
  Calibration progress: 300/300

[OK] Engine build successful!
     Serializing engine to: models/biomedclip/tensorrt/vision_encoder_int8.engine

======================================================================
BUILD COMPLETE
======================================================================
[OK] TensorRT engine saved: models/biomedclip/tensorrt/vision_encoder_int8.engine
     Engine size: 95.32 MB (vs ~330MB FP32 ONNX)
     Compression: 3.5x smaller
[OK] Calibration cache saved: models/biomedclip/tensorrt/calibration.cache

Next steps:
  1. Run validation: python3 tests/benchmark_tensorrt.py
  2. Test pipeline: streamlit run streamlit_app.py
======================================================================
```

### Troubleshooting

**Problem: Out of Memory (OOM) during build**
```bash
# Reduce workspace size
python3 src/models/build_tensorrt_engine.py --workspace 256

# Or even smaller
python3 src/models/build_tensorrt_engine.py --workspace 128
```

**Problem: Build takes >45 minutes**
- This is normal if you have many calibration images
- JPEG decoding is CPU-bound on Jetson Nano
- You can monitor CPU usage: `htop`

**Problem: ONNX parsing error**
- Check ONNX opset version (should be 13-16)
- Verify both `.onnx` and `.onnx.data` files are in the same directory

**Problem: No calibration images found**
```bash
# Check image directory
ls -lh data/calibration_images/

# If you have .jpeg instead of .jpg, the script will auto-detect
```

---

## Step 2: Validate INT8 Accuracy and Performance

**Estimated time:** 2-3 minutes

### Command

```bash
python3 tests/benchmark_tensorrt.py \
    --test-dir data/calibration_images \
    --warmup 10 \
    --iterations 100
```

### What to Expect

The benchmark will:
1. Load TensorRT INT8 engine
2. Load PyTorch FP32 model (for comparison)
3. Warm up the engine (10 iterations)
4. Measure latency (100 iterations)
5. Compare accuracy against PyTorch

**Console Output:**
```
======================================================================
TensorRT INT8 Benchmark and Validation
======================================================================
[OK] Running on Jetson: NVIDIA Jetson Nano

[1/4] Loading models...
  Loading TensorRT INT8 model...
[OK] TensorRT engine loaded: models/biomedclip/tensorrt/vision_encoder_int8.engine
     Engine size: 95.32 MB
     Text embeddings: (3, 512)
     Logit scale: 85.2323
  Loading PyTorch FP32 model...
Loading BiomedCLIP model...
[OK] Using 20 test images

[2/4] Loading test images from: data/calibration_images
[OK] Using 20 test images

[3/4] Warming up TensorRT engine (10 iterations)...
[OK] Warmup complete

[4/4] Benchmarking latency (100 iterations)...
  Progress: 25/100 (current: 245.32ms)
  Progress: 50/100 (current: 238.15ms)
  Progress: 75/100 (current: 252.44ms)
  Progress: 100/100 (current: 241.09ms)

[5/5] Comparing accuracy (TensorRT vs PyTorch)...

Image                          TensorRT   PyTorch    Diff       Match
----------------------------------------------------------------------
calib_0001.jpg                 OSCC       OSCC       0.0234     YES
calib_0002.jpg                 OPMD       OPMD       0.0156     YES
calib_0003.jpg                 Normal     Normal     0.0089     YES
...

======================================================================
BENCHMARK RESULTS
======================================================================

LATENCY (TensorRT INT8):
  Mean:       243.56 ms
  Median:     241.32 ms
  Std Dev:    18.43 ms
  Min:        215.67 ms
  Max:        298.12 ms
  P95:        275.45 ms
  P99:        289.34 ms
  Throughput: 4.11 FPS

[OK] Latency meets target: 243.56ms <= 350ms

ACCURACY (TensorRT vs PyTorch):
  Max score difference: 0.0342
  Class agreement:      10/10 (100.0%)

[OK] Accuracy acceptable: 0.0342 < 0.050

======================================================================
SUMMARY
======================================================================
[OK] TensorRT engine validated successfully!

Next steps:
  1. Test full pipeline: streamlit run streamlit_app.py
  2. Run extended stress test for thermal throttling
======================================================================
```

### Success Criteria

✓ **Latency:** Mean < 350ms (target: 200-350ms for ViT-B/16)
✓ **Accuracy:** Max score difference < 0.05 (5%)
✓ **Class Agreement:** >90% (ideally 100%)

### If Validation Fails

**High Latency (>350ms)**
- Check thermal throttling: `tegrastats`
- Monitor temperature: `cat /sys/devices/virtual/thermal/thermal_zone*/temp`
- This is expected for ViT-B/16 on Maxwell GPU - consider smaller model if critical

**Low Accuracy (<95% agreement)**
- Check calibration data quality (should be diverse and representative)
- Try rebuilding with more images (400+)
- Consider using FP16 engine instead (see Alternative section below)

---

## Step 3: Test Full OncoEdge Pipeline

**Estimated time:** 1 minute

### Command

```bash
# Start Streamlit app
streamlit run streamlit_app.py
```

### What to Expect

**Console Output:**
```
Initializing OncoEdge Pipeline...
[OK] Detected Jetson Nano with TensorRT engine
[OK] Loading TensorRT INT8 backend...
Loading TensorRT engine: models/biomedclip/tensorrt/vision_encoder_int8.engine
[OK] TensorRT engine loaded: models/biomedclip/tensorrt/vision_encoder_int8.engine
     Engine size: 95.32 MB
     Text embeddings: (3, 512)
     Logit scale: 85.2323
[OK] OncoEdge pipeline initialized (backend: tensorrt)

You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

### Testing the Pipeline

1. Open browser to `http://localhost:8501`
2. Upload a test image with oral lesion
3. Enter patient metadata (age, tobacco history, lesion duration)
4. Click "Analyze Image"
5. Verify:
   - YOLO detection boxes appear
   - BiomedCLIP classification works (OSCC/OPMD/Normal)
   - Risk assessment generates
   - No errors in console

**Pipeline should process image in 335-610ms:**
- YOLO: ~100-200ms
- BiomedCLIP (TensorRT): ~200-350ms
- Fusion + Risk: ~5-10ms
- Overhead: ~30-50ms

---

## Verification Checklist

After completing all steps, verify:

- [ ] TensorRT engine file exists (~70-120 MB)
- [ ] Engine loads without errors
- [ ] Benchmark latency: 200-350ms
- [ ] Benchmark accuracy: <5% difference from PyTorch
- [ ] Class agreement: >90%
- [ ] Pipeline auto-detects Jetson and uses TensorRT
- [ ] Full pipeline works end-to-end
- [ ] No OOM errors during inference
- [ ] Consistent performance (no thermal throttling in first 5 minutes)

---

## Alternative: FP16 Engine (Simpler, Faster Build)

If INT8 calibration is problematic or you want faster deployment, try FP16:

### Build FP16 Engine (2-3 minutes, no calibration)

```bash
# Using trtexec (simpler)
trtexec --onnx=models/biomedclip/onnx/vision_encoder.onnx \
        --fp16 \
        --workspace=512 \
        --saveEngine=models/biomedclip/tensorrt/vision_encoder_fp16.engine \
        --verbose
```

### Update Wrapper to Use FP16

Edit `src/models/biomedclip_tensorrt.py`, line 46:
```python
# Change default engine_path
engine_path='models/biomedclip/tensorrt/vision_encoder_fp16.engine',
```

### FP16 Tradeoffs

**Pros:**
- Much faster build: 2-3 minutes (vs 15-30 minutes)
- No calibration images needed
- No accuracy drift risk
- Simpler workflow

**Cons:**
- Slightly slower than INT8: ~1.3-1.5× speedup (vs 1.5-2× for INT8)
- Slightly larger: ~170MB (vs 70-120MB for INT8)

**Recommendation:** Try FP16 first. Only use INT8 if you need maximum performance.

---

## Performance Monitoring

### Monitor GPU Utilization

```bash
# Real-time stats
tegrastats

# Watch GPU frequency
watch -n 1 'cat /sys/devices/gpu.0/devfreq/*/cur_freq'
```

### Monitor Temperature

```bash
# Check current temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Watch temperature during inference
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'
```

### Thermal Throttling Check

Run sustained inference for 10 minutes and monitor latency:
```bash
python3 tests/benchmark_tensorrt.py --iterations 1000
```

If latency increases >20% after 5-10 minutes → thermal throttling
- Add cooling (heatsink, fan)
- Reduce ambient temperature

---

## Common Issues and Solutions

### Issue: Pipeline uses PyTorch instead of TensorRT

**Symptoms:**
```
[OK] Running on development machine
[OK] Loading PyTorch FP32 backend (device: cuda)...
```

**Solutions:**
1. Check engine exists: `ls models/biomedclip/tensorrt/vision_encoder_int8.engine`
2. Check Jetson detection: `cat /etc/nv_tegra_release`
3. Force TensorRT: Modify `streamlit_app.py` to pass `force_backend='tensorrt'`

### Issue: Import errors (pycuda, tensorrt)

```bash
# Install pycuda
pip3 install pycuda

# TensorRT should be pre-installed with JetPack
# If missing, check JetPack installation:
sudo apt update
sudo apt install nvidia-jetpack
```

### Issue: ONNX.data file not found

**Error:** TensorRT fails to load weights

**Solution:** Both files must be in same directory:
```bash
# Verify both exist
ls -lh models/biomedclip/onnx/vision_encoder.onnx
ls -lh models/biomedclip/onnx/vision_encoder.onnx.data
```

### Issue: Slow calibration (>45 minutes)

**This is normal if:**
- You have 400+ calibration images
- Jetson Nano CPU is busy with other tasks

**Speed up:**
- Use fewer images (250 minimum)
- Close other applications
- The slow part is JPEG decode (CPU-bound) - cannot be avoided

### Issue: Engine build fails with opset error

**Error:** `Unsupported ONNX opset version: 17`

**Solution:** ONNX export must use opset 13-16 for Jetson Nano
- Check your ONNX export script uses `opset_version=13`
- Re-export on Windows if needed

---

## Performance Expectations

### Realistic Targets (Jetson Nano 4GB)

| Metric | PyTorch FP32 | TensorRT INT8 | Improvement |
|--------|--------------|---------------|-------------|
| Model Size | 344 MB | 70-120 MB | 3-5× smaller |
| Memory Usage | ~1.8-2.2 GB | ~800MB-1.3GB | ~1.5-2× less |
| Vision Latency | 400-600 ms | 200-350 ms | 1.5-2× faster |
| Complete Pipeline | ~600-900 ms | 335-610 ms | **1.6-3 FPS** |

### Notes

- **Maxwell GPU:** No Tensor Cores, so INT8 gains come from memory bandwidth, not hardware acceleration
- **ViT-B/16:** Transformers are slower than CNNs on embedded devices
- **Thermal throttling:** Sustained FPS < burst FPS (expect 10-20% drop after 10 minutes)
- **Variability:** ±30% from these numbers is normal

---

## Next Steps After Deployment

1. **Stress Testing:**
   - Run 1000+ inferences to check thermal stability
   - Monitor GPU temperature and frequency
   - Verify no memory leaks

2. **Integration Testing:**
   - Test with real patient images
   - Verify clinical decision tree works
   - Test edge cases (very small lesions, poor lighting)

3. **Optimization (if needed):**
   - If latency still too high: Consider YOLO11n (smaller detector)
   - If memory tight: Switch to FP16 or smaller ViT model
   - If accuracy drops: Collect more calibration data

4. **Production Deployment:**
   - Setup auto-start on boot
   - Add monitoring/logging
   - Implement error handling/recovery

---

## Support

**Created files:**
- `src/models/build_tensorrt_engine.py` - Engine builder
- `src/models/biomedclip_tensorrt.py` - TensorRT inference wrapper
- `src/pipeline/inference_pipeline.py` - Auto-detection (modified)
- `tests/benchmark_tensorrt.py` - Validation script

**Reference:**
- Full quantization plan: `C:\Users\1aapa\.claude\plans\abstract-pondering-wand.md`
- Windows preparation: `QUANTIZATION_README.md`

**Issues?**
- Check console output for specific error messages
- Verify all prerequisites met
- Try FP16 engine if INT8 problematic
- Monitor system resources (memory, temperature, GPU frequency)

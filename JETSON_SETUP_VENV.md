# Jetson Virtual Environment Setup for TensorRT

## Problem
Your venv has all Python packages (torch, open-clip, etc.) but cannot import `tensorrt` or `pycuda` because they are system packages installed with JetPack.

## Solution: Use venv with system site-packages access

### Step 1: Enable system site-packages in your venv

```bash
# Activate your venv (from ~/OncoEdge/ directory)
cd ~/OncoEdge
source OncoEdge-Jetson/bin/activate

# Enable access to system site-packages
# This allows venv to see TensorRT while keeping your other packages isolated
echo "import sys; sys.path.append('/usr/lib/python3.10/dist-packages')" > OncoEdge-Jetson/lib/python3.10/site-packages/jetson_packages.pth
```

### Step 2: Install pycuda in venv

```bash
# Still in venv - use uv pip (your package manager)
uv pip install pycuda
```

### Step 3: Verify both imports work

```bash
python3 -c "import tensorrt as trt; import pycuda.autoinit; print(f'TensorRT: {trt.__version__}'); print('pycuda: OK')"
```

**Expected output:**
```
TensorRT: 10.3.0.26
pycuda: OK
```

### Step 4: Verify all other packages still work

```bash
python3 -c "import torch; import open_clip; print(f'PyTorch: {torch.__version__}'); print(f'open_clip: OK')"
```

## Now you can run the TensorRT build

```bash
# All in venv with access to both system packages (TensorRT) and venv packages (torch, open_clip)
cd /path/to/oncoedge
python3 src/models/build_tensorrt_engine.py
```

---

## Alternative: Use system Python (NOT recommended)

If above doesn't work, you can use system Python directly:

```bash
# Deactivate venv
deactivate

# Install packages in system Python (use uv or pip3)
sudo pip3 install torch torchvision open_clip_torch

# Run build
python3 src/models/build_tensorrt_engine.py
```

**Downside:** Pollutes system Python with project dependencies. Use venv approach instead.

---

## Why this works

- **TensorRT**: System package in `/usr/lib/python3.10/dist-packages` (installed by JetPack)
- **pycuda**: Can be installed via pip (compiles against system CUDA)
- **Your venv**: Adding `.pth` file makes system packages visible to venv
- **Result**: Best of both worlds - isolated project dependencies + system TensorRT access

## Verification checklist

Run these commands **in your venv**:

```bash
cd ~/OncoEdge
source OncoEdge-Jetson/bin/activate

# Test TensorRT
python3 -c "import tensorrt as trt; print(trt.__version__)"

# Test pycuda
python3 -c "import pycuda.driver as cuda; import pycuda.autoinit; print('pycuda OK')"

# Test your project dependencies
python3 -c "import torch; import open_clip; from PIL import Image; import numpy as np; print('All packages OK')"
```

If all three pass → You're ready to build the engine!

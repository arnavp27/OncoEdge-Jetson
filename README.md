# OncoEdge: Oral Cancer Detection System

AI-powered screening system for oral cancer detection using YOLO11n-seg and BiomedCLIP.

## Setup (Jetson Orin NX)

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and Install
```bash
git clone <repo-url>
cd oncoedge
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies with latest versions
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install ultralytics==8.4.8 open-clip-torch==3.2.0 streamlit==1.54.0 opencv-python==4.12.0.88 pillow==12.1.0 numpy pandas matplotlib
```

### Run Application
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Access at: `http://<jetson-ip>:8501`

## Usage

1. Enter patient information in sidebar
2. Upload oral cavity image
3. Click "Analyze Image"
4. Review risk assessment and detected lesions

## System Requirements

- NVIDIA Jetson Orin NX (16GB RAM)
- CUDA 11.8+
- Python 3.10

## Project Structure

See `claude.md` for detailed implementation guidelines.

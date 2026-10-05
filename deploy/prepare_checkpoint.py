"""Extract and verify the project's trained detector, without installing packages."""
import hashlib
from pathlib import Path
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_REVISION = "cf745d5785f9422659928701dca7bb56501db9c2"
CHECKPOINT_PATH = "zenodo_DOAOC/yolo_finetune/oncoedge_v1/weights/best.pt"
CHECKPOINT_SHA256 = "823cdd1190d18871b75b9f82a869c594184090a9c07c224719e666da6581f110"


def checkpoint_digest(path):
    with path.open("rb") as checkpoint:
        return hashlib.file_digest(checkpoint, "sha256").hexdigest()


def main():
    destination = PROJECT_ROOT / "models" / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checkpoint_digest(destination) != CHECKPOINT_SHA256:
            raise SystemExit(
                f"Checkpoint checksum mismatch: {destination}. "
                "Keep your existing file and supply the project's verified checkpoint."
            )
        print(f"Verified trained checkpoint: {destination}")
        return

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            result = subprocess.run(
                ["git", "show", f"{CHECKPOINT_REVISION}:{CHECKPOINT_PATH}"],
                cwd=PROJECT_ROOT,
                stdout=temporary,
                stderr=subprocess.PIPE,
            )
        if result.returncode:
            raise SystemExit(
                "The checkpoint's training commit is unavailable locally. "
                "Run 'git fetch origin develop_yolo_ft', then retry."
            )
        if checkpoint_digest(temporary_path) != CHECKPOINT_SHA256:
            raise SystemExit("Extracted checkpoint checksum mismatch; no model was installed.")
        temporary_path.chmod(0o644)
        temporary_path.replace(destination)
        print(f"Installed verified trained checkpoint: {destination}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

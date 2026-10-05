"""Regression checks for model reuse and checkpoint provisioning."""
import hashlib
from unittest.mock import Mock

import pytest

from deploy import prepare_checkpoint
from src.models import biomedclip_classifier


def test_cached_weights_use_local_model_and_upstream_tokenizer(tmp_path, monkeypatch):
    # OpenCLIP's local-dir tokenizer expects a vocabulary in that directory.
    # Our saved checkpoint has weights/config, so retain the upstream tokenizer.
    monkeypatch.setattr(biomedclip_classifier, "__file__", str(tmp_path / "src/models/classifier.py"))
    saved_model = tmp_path / "models/biomedclip"
    saved_model.mkdir(parents=True)
    (saved_model / "open_clip_pytorch_model.bin").write_bytes(b"cached weights")
    (saved_model / "open_clip_config.json").write_text("{}")
    download = Mock(side_effect=AssertionError("Cached assets must not download again"))
    create = Mock(return_value=(Mock(), None, Mock()))
    tokenizer = Mock()
    monkeypatch.setattr(biomedclip_classifier, "hf_hub_download", download)
    monkeypatch.setattr(biomedclip_classifier.open_clip, "create_model_and_transforms", create)
    monkeypatch.setattr(biomedclip_classifier.open_clip, "get_tokenizer", tokenizer)
    monkeypatch.setattr(biomedclip_classifier.BiomedCLIPClassifier, "_encode_text_prompts", Mock())

    biomedclip_classifier.BiomedCLIPClassifier(device="cpu")

    create.assert_called_once_with("local-dir:" + str(saved_model))
    tokenizer.assert_called_once_with("hf-hub:" + biomedclip_classifier.BiomedCLIPClassifier.MODEL_ID)
    download.assert_not_called()


def test_existing_weights_only_download_missing_architecture_config(tmp_path, monkeypatch):
    monkeypatch.setattr(biomedclip_classifier, "__file__", str(tmp_path / "src/models/classifier.py"))
    saved_model = tmp_path / "models/biomedclip"
    saved_model.mkdir(parents=True)
    checkpoint = saved_model / "open_clip_pytorch_model.bin"
    checkpoint.write_bytes(b"existing weights")
    requested_files = []

    def download(*, repo_id, filename, local_dir):
        requested_files.append(filename)
        (local_dir / filename).write_text("{}")

    monkeypatch.setattr(biomedclip_classifier, "hf_hub_download", download)
    classifier = biomedclip_classifier.BiomedCLIPClassifier.__new__(biomedclip_classifier.BiomedCLIPClassifier)

    assert classifier._ensure_local_model() == saved_model
    assert requested_files == ["open_clip_config.json"]
    assert checkpoint.read_bytes() == b"existing weights"


def test_checkpoint_preparation_preserves_unexpected_existing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_checkpoint, "PROJECT_ROOT", tmp_path)
    checkpoint = tmp_path / "models/best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"a user's different trained model")

    with pytest.raises(SystemExit, match="checksum mismatch"):
        prepare_checkpoint.main()

    assert checkpoint.read_bytes() == b"a user's different trained model"


def test_incomplete_git_artifact_is_never_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_checkpoint, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prepare_checkpoint, "CHECKPOINT_SHA256", hashlib.sha256(b"complete model").hexdigest())

    def incomplete_extraction(command, *, cwd, stdout, stderr):
        stdout.write(b"truncated")
        return Mock(returncode=0)

    monkeypatch.setattr(prepare_checkpoint.subprocess, "run", incomplete_extraction)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        prepare_checkpoint.main()

    assert not (tmp_path / "models/best.pt").exists()
    assert list((tmp_path / "models").iterdir()) == []

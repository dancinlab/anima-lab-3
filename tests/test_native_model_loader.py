from pathlib import Path

import pytest

import model_loader


def test_native_model_requires_complete_directory(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_loader, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        model_loader.load_model("anima-native")


def test_model_list_marks_native_only_when_both_files_exist(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_loader, "MODELS_DIR", tmp_path)
    model_dir = tmp_path / "anima-native"
    model_dir.mkdir()
    (model_dir / "final.pt").write_bytes(b"model")
    first = model_loader.list_available_models()[0]
    assert first[0] == "anima-native" and first[2] is False
    (model_dir / "tokenizer.json").write_text("{}")
    assert model_loader.list_available_models()[0][2] is True

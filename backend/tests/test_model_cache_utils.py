import os
from pathlib import Path

from services.model_cache_manager import ModelCacheValidator
import services.model_cache_manager as mcm


def test_get_directory_size_counts_files(tmp_path):
    root = tmp_path / "dir"
    root.mkdir()
    f1 = root / "a.txt"
    f1.write_bytes(b"abc")
    sub = root / "sub"
    sub.mkdir()
    f2 = sub / "b.txt"
    f2.write_bytes(b"de")

    size = ModelCacheValidator._get_directory_size(str(root))
    assert size == f1.stat().st_size + f2.stat().st_size


def test_get_file_hash_matches_expected(tmp_path):
    f = tmp_path / "file.bin"
    f.write_bytes(b"hello world")
    md5 = ModelCacheValidator._get_file_hash(str(f), "md5")
    assert md5 is not None and len(md5) == 32
    sha256 = ModelCacheValidator._get_file_hash(str(f), "sha256")
    assert sha256 is not None and len(sha256) == 64


def test_cache_uploaded_model_archive_unknown_model_type_raises(tmp_path):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    validator = ModelCacheValidator(str(hf_home))
    validator.configs = {}
    archive = tmp_path / "m.zip"
    archive.write_bytes(b"notazip")
    try:
        mcm.cache_uploaded_model_archive(str(hf_home), "no_such_model", str(archive), validator=validator)
        assert False, "Expected ValueError for unknown model type"
    except ValueError:
        pass

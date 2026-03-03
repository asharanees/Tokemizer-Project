import os
import shutil
from pathlib import Path

import services.model_cache_manager as mcm


def test_resolve_model_repo_coref_alias():
    repo = mcm._resolve_model_repo("coreference", mcm._COREF_MINILM_MODEL_ALIAS)
    assert repo == mcm._COREF_MINILM_MODEL_ID


def test_model_cache_dir_name_and_find(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    repo = "org/model-name"
    dirname = mcm._model_cache_dir_name(repo)
    # simulate a cached folder with additional hash suffix
    folder = cache / f"{dirname}-12345"
    folder.mkdir()
    found = mcm._find_model_path_in_cache(str(cache), repo)
    assert found is not None
    assert os.path.basename(found).startswith(dirname)


def test_cleanup_stale_model_dirs_removes_others(tmp_path):
    cache = tmp_path / "hub"
    cache.mkdir()
    repo = "org/model"
    keep = cache / mcm._model_cache_dir_name(repo)
    keep.mkdir()
    stale = cache / (mcm._model_cache_dir_name(repo) + "-old")
    stale.mkdir()
    # ensure stale exists
    assert stale.exists()
    mcm._cleanup_stale_model_dirs(str(cache), repo, keep_path=str(keep))
    assert keep.exists()
    assert not stale.exists()

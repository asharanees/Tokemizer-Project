import services.optimizer.core as optimizer_core
from services.optimizer.core import PromptOptimizer


class _FakeNLP:
    def __init__(self, pipe_names):
        self.pipe_names = list(pipe_names)

    def disable_pipes(self, *pipes):
        self.pipe_names = [pipe for pipe in self.pipe_names if pipe not in pipes]


class _FakeSpacy:
    def __init__(self):
        self.calls = 0

    def load(self, _target):
        self.calls += 1
        return _FakeNLP(["tok2vec", "tagger", "parser", "ner", "senter"])


def _model_configs():
    return {
        "semantic_guard": {"model_name": "test-semantic"},
        "semantic_rank": {"model_name": "test-semantic-rank"},
        "token_classifier": {"model_name": "test-token-classifier"},
        "coreference": {"model_name": "test-coref"},
    }


def test_semantic_and_linguistic_nlp_loaders_are_independent(monkeypatch):
    fake_spacy = _FakeSpacy()
    monkeypatch.setattr(optimizer_core, "_import_spacy", lambda: fake_spacy)
    monkeypatch.setattr(optimizer_core, "get_model_configs", _model_configs)

    optimizer = PromptOptimizer()

    semantic_nlp = optimizer._get_nlp_model()
    linguistic_nlp = optimizer._get_linguistic_nlp_model()

    assert semantic_nlp is not None
    assert linguistic_nlp is not None
    assert semantic_nlp is not linguistic_nlp

    # Semantic model disables heavy linguistic pipes for lightweight dedup/similarity.
    assert "parser" not in semantic_nlp.pipe_names
    assert "ner" not in semantic_nlp.pipe_names

    # Linguistic model keeps parser/NER behavior for linguistic trimming.
    assert "parser" in linguistic_nlp.pipe_names
    assert "ner" in linguistic_nlp.pipe_names


def test_refresh_model_configs_clears_nlp_caches_and_reloads(monkeypatch):
    fake_spacy = _FakeSpacy()
    monkeypatch.setattr(optimizer_core, "_import_spacy", lambda: fake_spacy)
    monkeypatch.setattr(optimizer_core, "get_model_configs", _model_configs)

    optimizer = PromptOptimizer()

    first_model = optimizer._get_nlp_model()
    assert first_model is not None
    assert fake_spacy.calls == 1

    optimizer.refresh_model_configs()

    # Refresh should clear cached semantic/linguistic spaCy state.
    assert optimizer._nlp is None
    assert optimizer._nlp_pipe_names == []
    assert optimizer._nlp_disabled_pipes == []
    assert optimizer._linguistic_nlp is None
    assert optimizer._linguistic_nlp_pipe_names == []

    second_model = optimizer._get_nlp_model()
    assert second_model is not None
    assert fake_spacy.calls == 2


def test_get_nlp_model_uses_nested_spacy_cache_dir(monkeypatch, tmp_path):
    model_root = tmp_path / "spacy" / "en_core_web_md"
    nested = model_root / "en_core_web_md-3.8.0"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "config.cfg").write_text("[nlp]\nlang = \"en\"\n", encoding="utf-8")

    class _PathAwareSpacy:
        def __init__(self):
            self.targets = []

        def load(self, target):
            self.targets.append(str(target))
            if str(target) != str(nested):
                raise OSError(f"unexpected target: {target}")
            return _FakeNLP(["tok2vec", "tagger", "parser", "ner", "senter"])

    fake_spacy = _PathAwareSpacy()
    monkeypatch.setattr(optimizer_core, "_import_spacy", lambda: fake_spacy)
    monkeypatch.setattr(optimizer_core, "get_model_configs", _model_configs)
    monkeypatch.setenv("PROMPT_OPTIMIZER_SPACY_MODEL_PATH", str(model_root))

    optimizer = PromptOptimizer()
    semantic_nlp = optimizer._get_nlp_model()

    assert semantic_nlp is not None
    assert fake_spacy.targets == [str(nested)]


def test_probe_semantic_rank_readiness_does_not_alias_guard_when_inventory_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        optimizer_core,
        "get_model_configs",
        lambda: {
            "semantic_guard": {"model_name": "test-semantic"},
            "token_classifier": {"model_name": "test-token-classifier"},
            "coreference": {"model_name": "test-coref"},
        },
    )

    def fake_similarity(_a, _b, _model_name, model_type="semantic_guard"):
        return 0.99 if model_type == "semantic_guard" else None

    monkeypatch.setattr(optimizer_core._metrics, "warm_up", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_core._metrics, "score_similarity", fake_similarity)

    optimizer = PromptOptimizer()

    guard_status = optimizer.probe_model_readiness("semantic_guard")
    rank_status = optimizer.probe_model_readiness("semantic_rank")

    assert guard_status["loaded"] is True
    assert rank_status["name"] != "test-semantic"
    assert rank_status["loaded"] is False


def test_probe_semantic_rank_readiness_stays_independent_when_configured(
    monkeypatch,
):
    monkeypatch.setattr(optimizer_core, "get_model_configs", _model_configs)

    calls = []

    def fake_similarity(_a, _b, model_name, model_type="semantic_guard"):
        calls.append((model_name, model_type))
        if model_type == "semantic_rank" and model_name == "test-semantic-rank":
            return None
        return 0.99

    monkeypatch.setattr(optimizer_core._metrics, "warm_up", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_core._metrics, "score_similarity", fake_similarity)

    optimizer = PromptOptimizer()
    optimizer._model_load_status = {
        "semantic_guard": {"name": "test-semantic", "loaded": True}
    }

    rank_status = optimizer.probe_model_readiness("semantic_rank")

    assert rank_status["name"] == "test-semantic-rank"
    assert rank_status["loaded"] is False
    assert ("test-semantic-rank", "semantic_rank") in calls


def test_warm_up_probes_semantic_rank_when_guard_missing(monkeypatch):
    monkeypatch.setattr(
        optimizer_core,
        "get_model_configs",
        lambda: {
            "semantic_rank": {"model_name": "test-semantic-rank"},
            "token_classifier": {"model_name": "test-token-classifier"},
            "coreference": {"model_name": "test-coref"},
        },
    )
    monkeypatch.setattr(PromptOptimizer, "_get_nlp_model", lambda self: None)
    monkeypatch.setattr(PromptOptimizer, "_get_coref_model", lambda self: None)
    monkeypatch.setattr(optimizer_core._entropy, "_get_fast_scorer", lambda: type("S", (), {"available": False})())
    monkeypatch.setattr(optimizer_core._entropy, "_get_scorer", lambda: type("S", (), {"available": False})())
    monkeypatch.setattr(
        optimizer_core._token_classifier,
        "_get_classifier",
        lambda _name: type("S", (), {"available": False})(),
    )

    calls = []

    def fake_warm_up(model_name, model_type="semantic_guard"):
        calls.append(("warm_up", model_name, model_type))
        if model_name is None:
            raise RuntimeError("missing model")

    def fake_similarity(_a, _b, model_name, model_type="semantic_guard"):
        calls.append(("score", model_name, model_type))
        if model_type == "semantic_rank" and model_name == "test-semantic-rank":
            return 0.91
        return None

    monkeypatch.setattr(optimizer_core._metrics, "warm_up", fake_warm_up)
    monkeypatch.setattr(optimizer_core._metrics, "score_similarity", fake_similarity)

    optimizer = PromptOptimizer()
    optimizer.warm_up()

    status = optimizer.model_status()
    assert status["semantic_guard"]["loaded"] is False
    assert status["semantic_rank"]["name"] == "test-semantic-rank"
    assert status["semantic_rank"]["loaded"] is True
    assert ("score", "test-semantic-rank", "semantic_rank") in calls


def test_warm_up_entropy_teacher_does_not_alias_fast_backend(monkeypatch):
    monkeypatch.setattr(optimizer_core, "get_model_configs", _model_configs)
    monkeypatch.setattr(PromptOptimizer, "_get_nlp_model", lambda self: None)
    monkeypatch.setattr(PromptOptimizer, "_get_coref_model", lambda self: None)
    monkeypatch.setattr(
        optimizer_core._entropy,
        "_get_fast_scorer",
        lambda: type("S", (), {"available": True})(),
    )
    monkeypatch.setattr(
        optimizer_core._entropy,
        "_get_scorer",
        lambda: type("S", (), {"available": False})(),
    )
    monkeypatch.setattr(optimizer_core._metrics, "warm_up", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        optimizer_core._metrics,
        "score_similarity",
        lambda *_args, **_kwargs: 0.99,
    )
    monkeypatch.setattr(
        optimizer_core._token_classifier,
        "_get_classifier",
        lambda _name: type("S", (), {"available": False})(),
    )

    optimizer = PromptOptimizer()
    optimizer.warm_up()

    status = optimizer.model_status()
    assert status["entropy_fast"]["loaded"] is True
    assert status["entropy"]["loaded"] is False


def test_probe_entropy_readiness_does_not_alias_fast_backend(monkeypatch):
    monkeypatch.setattr(
        optimizer_core._entropy,
        "_get_fast_scorer",
        lambda: type("S", (), {"available": True})(),
    )
    monkeypatch.setattr(
        optimizer_core._entropy,
        "_get_scorer",
        lambda: type("S", (), {"available": False})(),
    )

    optimizer = PromptOptimizer()

    fast_status = optimizer.probe_model_readiness("entropy_fast")
    teacher_status = optimizer.probe_model_readiness("entropy")

    assert fast_status["loaded"] is True
    assert teacher_status["loaded"] is False


def test_probe_semantic_rank_does_not_alias_guard_when_cache_empty(monkeypatch):
    monkeypatch.setattr(
        optimizer_core,
        "get_model_configs",
        lambda: {
            "semantic_guard": {"model_name": "test-semantic"},
            "token_classifier": {"model_name": "test-token-classifier"},
            "coreference": {"model_name": "test-coref"},
        },
    )

    def fake_similarity(_a, _b, model_name, model_type="semantic_guard"):
        if model_type == "semantic_rank":
            return None
        if model_type == "semantic_guard" and model_name == "test-semantic":
            return 0.95
        return None

    monkeypatch.setattr(optimizer_core._metrics, "warm_up", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_core._metrics, "score_similarity", fake_similarity)

    optimizer = PromptOptimizer()
    optimizer._model_load_status = {}

    rank_status = optimizer.probe_model_readiness("semantic_rank")

    assert rank_status["name"] != "test-semantic"
    assert rank_status["loaded"] is False


def test_coref_loader_maps_repo_id_to_registered_spacy_pipe(monkeypatch):
    class _CorefSpacyPipeline:
        def __init__(self):
            self.pipe_names = []
            self.added = []

        def add_pipe(self, name):
            self.added.append(name)
            self.pipe_names.append(name)

    class _CorefSpacy:
        def __init__(self):
            self.pipeline = _CorefSpacyPipeline()

        def blank(self, _lang):
            return self.pipeline

    monkeypatch.setattr(
        optimizer_core,
        "get_model_configs",
        lambda: {
            "semantic_guard": {"model_name": "test-semantic"},
            "semantic_rank": {"model_name": "test-semantic-rank"},
            "token_classifier": {"model_name": "test-token-classifier"},
            "coreference": {"model_name": optimizer_core._COREF_MODEL_NAME},
        },
    )
    fake_spacy = _CorefSpacy()
    monkeypatch.setattr(optimizer_core, "_import_spacy", lambda: fake_spacy)
    monkeypatch.setattr(optimizer_core, "_import_spacy_coref", lambda: object())

    optimizer = PromptOptimizer()
    coref_model = optimizer._get_coref_model()

    assert coref_model is fake_spacy.pipeline
    assert fake_spacy.pipeline.added == ["coref_minilm"]
    assert optimizer._coref_pipe_name == "coref_minilm"

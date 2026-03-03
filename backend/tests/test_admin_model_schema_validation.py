import pytest
from pydantic import ValidationError

from routers.admin_routes import ModelCreate, ModelUpdate


def test_model_create_rejects_negative_min_size_bytes():
    with pytest.raises(ValidationError):
        ModelCreate(model_type="custom", model_name="org/model", min_size_bytes=-1)


def test_model_update_rejects_negative_min_size_bytes():
    with pytest.raises(ValidationError):
        ModelUpdate(model_name="org/model", min_size_bytes=-10)

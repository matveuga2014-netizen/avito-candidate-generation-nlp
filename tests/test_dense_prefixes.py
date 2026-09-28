import pytest

from src.channels.dense import E5_PREFIXES, NO_PREFIXES, prefixes


def test_known_models():
    assert prefixes("artifacts/e5-base-ft") == E5_PREFIXES == ("query: ", "passage: ")
    assert prefixes("artifacts/bge-m3-ft") == NO_PREFIXES == ("", "")


def test_unknown_model_is_error():
    with pytest.raises(KeyError):
        prefixes("/some/path/e5-like-name")   # раньше подстрока «e5» молча дала бы префиксы E5

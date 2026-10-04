import math

import pytest

from ragqa.embedders import HashingEmbedder, SentenceTransformerEmbedder, get_embedder


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_vectors_have_the_right_size_and_unit_length():
    embedder = HashingEmbedder(dimension=64)
    vectors = embedder.embed(["backups run nightly", "vacation days"])

    assert all(len(v) == 64 for v in vectors)
    assert all(math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-9) for v in vectors)


def test_same_text_gives_same_vector():
    embedder = HashingEmbedder()
    assert embedder.embed(["pump inspection"]) == embedder.embed(["pump inspection"])


def test_related_text_is_closer_than_unrelated_text():
    embedder = HashingEmbedder()
    query, related, unrelated = embedder.embed(
        [
            "how many vacation days do employees get",
            "employees receive thirty vacation days per year",
            "the pump seals are replaced every year",
        ]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_text_without_words_still_gives_a_usable_vector():
    vector = HashingEmbedder().embed(["... !!!"])[0]
    assert math.isclose(sum(x * x for x in vector), 1.0)


def test_unknown_embedder_name_raises():
    with pytest.raises(ValueError):
        get_embedder("nope")


def test_sentence_transformer_wrapper_uses_the_given_model():
    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            return [[1.0, 0.0, 0.0] for _ in texts]

    embedder = SentenceTransformerEmbedder("org/tiny-model", model=FakeModel())

    assert embedder.dimension == 3
    assert embedder.name == "tiny-model-3"
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_empty_embedder_setting_falls_back_to_hashing(monkeypatch):
    monkeypatch.setenv("RAGQA_EMBEDDER", "")
    assert get_embedder().name == "hashing-384"

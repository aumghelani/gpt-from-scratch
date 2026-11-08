import pytest
from fastapi.testclient import TestClient

from gpt_scratch.cli import main
from gpt_scratch.serve.app import create_app
from gpt_scratch.serve.engine import Engine

TEXT = "\n\n".join(f"SPEAKER{i % 3}:\nto be or not\nto be that is\nthe question now" for i in range(60))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("multi")
    src = root / "in.txt"
    src.write_text(TEXT, encoding="utf-8")
    for tokenizer in ("char", "bpe"):
        data, run = root / f"data-{tokenizer}", root / f"{tokenizer}-model"
        assert main(["prepare", "--input", str(src), "--out", str(data), "--tokenizer", tokenizer,
                     "--vocab-size", "300"]) == 0
        assert main(["train", "--data", str(data), "--out", str(run), "--preset", "test", "--device", "cpu"]) == 0
    assert main(["finetune", "--run", str(root / "bpe-model"), "--from-text", str(src),
                 "--out", str(root / "chat-model"), "--steps", "4", "--device", "cpu"]) == 0
    return {"char": root / "char-model", "chat": root / "chat-model"}


@pytest.fixture(scope="module")
def client(runs):
    return TestClient(create_app([runs["char"], runs["chat"]], device="cpu"))


def test_models_endpoint_lists_every_run_with_its_mode(client):
    models = client.get("/v1/models").json()
    assert [m["id"] for m in models] == ["char-model", "chat-model"]
    assert [m["chat"] for m in models] == [False, True]
    assert all(m["params"] > 0 and m["block_size"] > 0 and m["checkpoint"].endswith(".pt") for m in models)


def test_the_first_run_is_the_default_model(client):
    assert client.get("/health").json()["model"] == "char-model"
    r = client.post("/v1/generate", json={"prompt": "to be", "max_new_tokens": 5, "temperature": 0})
    assert r.status_code == 200


def test_a_request_can_choose_a_model(client):
    r = client.post("/v1/chat", json={"model": "chat-model", "max_new_tokens": 5, "temperature": 0,
                                     "messages": [{"role": "user", "content": "to be"}]})
    assert r.status_code == 200 and "text" in r.json()
    # the default (char) model cannot chat, so the same request without `model` is refused
    r = client.post("/v1/chat", json={"max_new_tokens": 5, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400


def test_an_unknown_model_is_a_404_that_lists_what_exists(client):
    r = client.post("/v1/generate", json={"model": "nope", "prompt": "to be"})
    assert r.status_code == 404 and "char-model" in r.json()["detail"]


def test_model_ids_must_be_unique(runs, tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        create_app([runs["char"], runs["char"]], device="cpu")


def test_at_least_one_run_is_required():
    with pytest.raises(ValueError):
        create_app([], device="cpu")


def test_engine_loads_the_best_checkpoint_by_default_and_can_load_the_latest(runs):
    assert Engine(runs["char"], "cpu").checkpoint_name == "best.pt"
    assert Engine(runs["char"], "cpu", checkpoint="latest").checkpoint_name == "latest.pt"


def test_ui_has_a_model_selector_filled_without_innerhtml(client):
    html = client.get("/").text
    assert 'id="model"' in html and "/v1/models" in html and "innerHTML" not in html


def test_serve_accepts_several_runs_and_a_checkpoint_choice(monkeypatch, runs):
    import uvicorn

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, **kw: captured.update(app=app))
    assert main(["serve", "--run", str(runs["char"]), str(runs["chat"]), "--checkpoint", "latest",
                 "--device", "cpu"]) == 0
    models = TestClient(captured["app"]).get("/v1/models").json()
    assert len(models) == 2 and {m["checkpoint"] for m in models} == {"latest.pt"}

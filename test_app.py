import pytest
from fastapi.testclient import TestClient

# ---------- Test 1: unit test generate_text with a fake pipe ----------
def test_generate_text_with_fake_pipe(monkeypatch):
    from models import generate_text  # uses your real function

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            # return any prompt-like string; generate_text just passes it to the pipe
            return "PROMPT"

    class FakePipe:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
        def __call__(self, prompt, temperature, max_new_tokens, do_sample, top_k, top_p):
            # Mimic HF pipeline output shape
            return [{"generated_text": "ignored</s>\n<|assistant|>\nHello world!"}]

    out = generate_text(FakePipe(), "any question")
    assert "Hello" in out


# “¿Qué pasa si no mando el prompt?”
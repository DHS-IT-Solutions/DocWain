import json
import tempfile
from pathlib import Path


def test_rebrand_config():
    from src.finetune.sprint.converter import rebrand_model_config

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(json.dumps({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "_name_or_path": "Qwen/Qwen3-14B",
        }))

        rebrand_model_config(Path(tmpdir), "DocWain-14B-v2")

        config = json.loads(config_path.read_text())
        assert config["_name_or_path"] == "DocWain-14B-v2"
        assert config["model_name"] == "DocWain-14B-v2"


def test_rebrand_tokenizer_config():
    from src.finetune.sprint.converter import rebrand_tokenizer_config

    with tempfile.TemporaryDirectory() as tmpdir:
        tok_path = Path(tmpdir) / "tokenizer_config.json"
        tok_path.write_text(json.dumps({
            "model_name": "Qwen3-14B",
            "chat_template": "some template",
        }))

        rebrand_tokenizer_config(Path(tmpdir), "DocWain-14B-v2")

        tok = json.loads(tok_path.read_text())
        assert tok["model_name"] == "DocWain-14B-v2"


def test_generate_model_card():
    from src.finetune.sprint.converter import generate_model_card

    card = generate_model_card("DocWain-14B-v2", {"accuracy": 4.5, "completeness": 4.2})
    assert "DocWain-14B-v2" in card
    assert "accuracy" in card.lower()

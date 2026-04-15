import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("sprint.converter")


def rebrand_model_config(model_dir: Path, model_name: str):
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    config["_name_or_path"] = model_name
    config["model_name"] = model_name
    config_path.write_text(json.dumps(config, indent=2))
    logger.info(f"Rebranded config.json → {model_name}")


def rebrand_tokenizer_config(model_dir: Path, model_name: str):
    tok_path = model_dir / "tokenizer_config.json"
    if not tok_path.exists():
        return
    tok = json.loads(tok_path.read_text())
    tok["model_name"] = model_name
    tok_path.write_text(json.dumps(tok, indent=2))
    logger.info(f"Rebranded tokenizer_config.json → {model_name}")


def generate_model_card(model_name: str, scores: dict) -> str:
    scores_md = "\n".join(f"| {k} | {v:.1f} |" for k, v in scores.items())
    return f"""---
license: apache-2.0
tags:
- document-intelligence
- enterprise
- extraction
- ocr
---

# {model_name}

Enterprise document intelligence model by DHS IT Solutions. Extracts, analyzes, and reasons about any document type with high accuracy.

## Evaluation Scores

| Dimension | Score (1-5) |
|-----------|-------------|
{scores_md}

## Capabilities

- Extraction from any document type (PDF, DOCX, Excel, CSV, images, scanned)
- Domain-aware reasoning across 8 enterprise domains
- Cross-document intelligence (comparison, aggregation, contradiction detection)
- Content generation grounded in document evidence
- OCR with degraded scan handling
- Hallucination-resistant with uncertainty flagging

## Usage

```python
from vllm import LLM
llm = LLM(model="{model_name}")
```
"""


def convert_to_base_model(checkpoint_dir: Path, output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / model_name

    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(checkpoint_dir, final_dir)

    rebrand_model_config(final_dir, model_name)
    rebrand_tokenizer_config(final_dir, model_name)

    logger.info(f"Base model ready at {final_dir}")
    return final_dir


def export_gguf(model_dir: Path, output_path: Path, quantization: str = "Q4_K_M") -> Optional[Path]:
    try:
        convert_script = "python -m llama_cpp.convert"
        cmd = f"{convert_script} {model_dir} --outfile {output_path} --outtype {quantization}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info(f"GGUF export complete: {output_path}")
            return output_path
        else:
            logger.warning(f"GGUF export failed: {result.stderr[:500]}")
            return None
    except Exception as e:
        logger.warning(f"GGUF export error: {e}")
        return None


def upload_to_huggingface(model_dir: Path, repo_id: str, model_card: str):
    try:
        from huggingface_hub import HfApi

        readme = model_dir / "README.md"
        readme.write_text(model_card)

        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(model_dir), repo_id=repo_id)
        logger.info(f"Uploaded to HuggingFace: {repo_id}")
    except ImportError:
        logger.warning("huggingface_hub not installed — skipping upload")
    except Exception as e:
        logger.warning(f"HuggingFace upload error: {e}")

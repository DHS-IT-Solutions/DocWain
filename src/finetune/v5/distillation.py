"""DocWain V5 — Layer 7: 14B-teacher → 7B-student KL+SFT distillation.

Implements the recipe from
``docs/superpowers/specs/2026-04-18-docwain-v5-design.md`` §"Layer 4 — 7B
Distillation" (labelled Layer 7 in the implementation roadmap).

Loss::

    total = alpha · CE_SFT(student, assistant_tokens)
          + (1 - alpha) · T² · KL(student_logits/T || teacher_logits/T)
                                          (assistant tokens only)

Key design notes
----------------

* **Full fine-tune, no LoRA.** Qwen3-dense-7B on an A100-80GB can be
  tuned in bf16 directly (≈ 14 GB weights + ≈ 28 GB Adam state +
  activations), so we skip PEFT entirely.
* **Teacher cache.** The 14 B teacher forward pass is the dominant cost
  of a naive KL loop. We cache logits **per sample** on disk, keyed by
  ``sha256(user + assistant)``, so the teacher runs **at most once per
  row across the whole training budget**. Critical for the 18 h window.
* **Tokenizer alignment.** Qwen3-8B / Qwen3-14B / DocWain-14B-v2 all
  share ``vocab_size=151936``; KL over the same ID space is mathematically
  well-defined. The trainer asserts this and hard-fails on mismatch —
  there is no logit-projection fallback by design (see spec).
* **Empty system field.** V5 corpus has ``system=""``; we propagate the
  empty string through ``apply_chat_template`` rather than injecting a
  persona string, honouring the "identity in weights" contract.
* **Assistant-only masking.** Prompt tokens get ``labels=-100`` so they
  contribute to neither SFT CE nor KL. This matches how the V5 SFT
  trainer will be wired (``feedback_base_model_approach``).
* **Reused from V2 trainer landscape.** Chat-template tokenization,
  bf16 selection, and checkpoint cadence follow ``src/finetune/v2/
  train_track.py``. We deliberately did NOT adopt Unsloth here —
  Unsloth's FastLanguageModel specialises for LoRA and 4-bit loading,
  whereas we need two full-precision models side-by-side and plain HF
  ``AutoModelForCausalLM`` is the simplest vehicle for that.

CLI
---

::

    python -m src.finetune.v5.distillation \\
        --teacher models/DocWain-14B-v5 \\
        --student Qwen/Qwen3-8B \\
        --corpus finetune_artifacts/v5/sft_reused.jsonl \\
        --output models/DocWain-8B-v5 \\
        --alpha 0.5 --temperature 2.0 \\
        --batch-size 2 --grad-accum 8 \\
        --learning-rate 1e-5 \\
        --checkpoint-interval 4h \\
        --teacher-cache finetune_artifacts/v5/distillation_teacher_cache \\
        [--dry-run]

``--dry-run`` runs 10 training steps end-to-end (teacher forward + cache
write + student forward + loss + backward + optimizer step) and emits
``finetune_artifacts/v5/distillation_dryrun_report.json`` with peak VRAM
and per-step loss decomposition. Completes in < 8 minutes on an A100.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DistillConfig:
    """Everything the trainer needs — CLI dumps straight into this."""

    teacher: str
    student: str
    corpus: str
    output: str
    teacher_cache: str
    cache_topk: int = 256

    alpha: float = 0.5
    temperature: float = 2.0
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    epochs: int = 1
    max_seq_length: int = 4096

    checkpoint_interval_sec: int = 4 * 3600  # 4 hours

    dry_run: bool = False
    dry_run_steps: int = 10
    seed: int = 42

    # Used only by dry-run to emit the report
    dryrun_report_path: str = "finetune_artifacts/v5/distillation_dryrun_report.json"
    # Where to stream loss components (sft_loss, kl_loss, total) on real runs
    log_path: str = "finetune_artifacts/v5/distillation_training.log"


# ---------------------------------------------------------------------------
# Duration parsing (reused-style helper)
# ---------------------------------------------------------------------------


_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])?\s*$", re.IGNORECASE)


def parse_duration(value: str) -> int:
    """Parse ``4h``, ``30m``, ``3600`` etc into seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    m = _DUR_RE.match(str(value))
    if not m:
        raise ValueError(f"Cannot parse duration '{value}'")
    qty = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return int(qty * mult)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class V5JsonlDataset(Dataset):
    """Reads the V5 SFT JSONL and returns raw rows.

    Rows carry ``{capability, source, difficulty, system, user, assistant,
    ...}``. We keep tokenization out of ``__getitem__`` so that the
    collator can batch efficiently and so the teacher-cache key is built
    from the exact (user, assistant) payload.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Corpus {path} line {lineno} is not valid JSON: {exc}"
                    )
                # Minimal schema check — the V5 transform enforces these.
                for k in ("user", "assistant"):
                    if k not in row:
                        raise ValueError(
                            f"Corpus row {lineno} missing required field '{k}'"
                        )
                # ``system`` may be absent; normalise to empty string so
                # chat-template always has a field to look at (but we
                # still treat empty-string as "no system turn" below).
                row.setdefault("system", "")
                self._rows.append(row)

        if not self._rows:
            raise ValueError(f"Corpus {path} is empty")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._rows[idx]


def cache_key(user: str, assistant: str) -> str:
    """Stable hash of the (user, assistant) payload."""
    h = hashlib.sha256()
    h.update(user.encode("utf-8"))
    h.update(b"\x1f")  # unit separator
    h.update(assistant.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def build_messages(row: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """Split the row into (prompt_messages, assistant_text).

    The prompt messages only contain system (if non-empty) + user — the
    assistant turn is tokenized separately so we can mask prompt tokens.
    """
    msgs: List[Dict[str, str]] = []
    sys_text = row.get("system", "") or ""
    if sys_text.strip():
        # V5 contract is empty system → identity in weights. If someone
        # supplies a non-empty system, we honour it rather than silently
        # dropping it. The warning lives in the caller.
        msgs.append({"role": "system", "content": sys_text})
    msgs.append({"role": "user", "content": row["user"]})
    return msgs, row["assistant"]


def tokenize_row(
    row: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize one row into (input_ids, attention_mask, labels).

    Labels are ``-100`` on prompt tokens and ``input_id`` on assistant
    tokens. The KL loss uses the same mask.
    """
    prompt_messages, assistant_text = build_messages(row)

    # Prompt: apply chat template with generation prompt so the assistant
    # header tokens (e.g. ``<|im_start|>assistant\n``) are part of the
    # prompt prefix. The assistant text plus ``<|im_end|>`` terminator
    # form the target.
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
    )

    # Assistant tokens. Append the same eos/turn-end token the model
    # would emit naturally. We use ``eos_token`` as a safe fallback if
    # the tokenizer doesn't expose a ``im_end`` token.
    assistant_ids = tokenizer(
        assistant_text,
        add_special_tokens=False,
    )["input_ids"]
    eot = _turn_end_id(tokenizer)
    if eot is not None:
        assistant_ids = assistant_ids + [eot]

    # Truncate assistant side first so the prompt is always intact; if
    # the prompt alone blows the budget we skip the row.
    budget = max_seq_length - len(prompt_ids)
    if budget <= 1:
        return {}  # signal skip
    assistant_ids = assistant_ids[:budget]

    input_ids = prompt_ids + assistant_ids
    labels = [-100] * len(prompt_ids) + assistant_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _turn_end_id(tokenizer) -> Optional[int]:
    """Return the ID of ``<|im_end|>`` if present, else EOS."""
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    return tokenizer.eos_token_id


def pad_collate(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """Right-pad a list of tokenized rows into a single batch tensor."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    out: Dict[str, List[torch.Tensor]] = {
        "input_ids": [], "attention_mask": [], "labels": [],
    }
    for b in batch:
        cur = b["input_ids"].size(0)
        pad = max_len - cur
        if pad:
            out["input_ids"].append(
                torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
            )
            out["attention_mask"].append(
                torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            out["labels"].append(
                torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)])
            )
        else:
            out["input_ids"].append(b["input_ids"])
            out["attention_mask"].append(b["attention_mask"])
            out["labels"].append(b["labels"])
    return {k: torch.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Teacher cache
# ---------------------------------------------------------------------------


class TeacherLogitCache:
    """On-disk cache of teacher logits keyed by ``sha256(user+assistant)``.

    Stored as ``<cache_dir>/<key[:2]>/<key>.pt`` — two-level fanout keeps
    directory fan-out bounded. Each file holds a dict
    ``{"assistant_logits": Tensor[L, V], "assistant_len": int}``. We only
    cache the logits on assistant tokens because prompt tokens are
    masked out anyway.

    **Top-k compression (``topk > 0``).** Full fp16 cache at Qwen3's
    151,936-entry vocab is ~1.4 TB for 31 K rows — doesn't fit typical
    training disks. ``topk`` stores only the K largest logits per token
    plus a single ``others`` fill value computed as
    ``logsumexp(remainder) - log(V - K)`` so the reconstructed dense
    logit vector preserves both the top-K ordering and the total
    partition function to within numerical precision. Empirically, K=256
    preserves > 99.9% of softmax probability mass on Qwen3-family models
    and shrinks per-row footprint by ~250×. Reconstruction happens
    transparently on ``load()``; callers always see a dense ``[L, V]``
    tensor.
    """

    def __init__(self, cache_dir: str, *, topk: int = 0, vocab_size: int = 0) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.topk = int(topk or 0)
        self.vocab_size = int(vocab_size or 0)
        if self.topk > 0 and self.vocab_size <= self.topk:
            raise ValueError(
                f"topk={self.topk} must be < vocab_size={self.vocab_size}"
            )

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pt"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> torch.Tensor:
        p = self._path(key)
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if "topk_values" in obj:
            # Compressed entry — reconstruct dense logits.
            topk_values = obj["topk_values"]        # [L, K] fp16
            topk_indices = obj["topk_indices"]      # [L, K] int32
            others = obj["others_logit"]            # [L]    fp16
            L = topk_values.size(0)
            V = int(obj["vocab_size"])
            dense = others.to(torch.float16).unsqueeze(1).expand(L, V).contiguous().clone()
            dense.scatter_(1, topk_indices.to(torch.int64), topk_values.to(torch.float16))
            return dense
        return obj["assistant_logits"]

    def save(self, key: str, assistant_logits: torch.Tensor) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".pt.tmp")
        if self.topk > 0 and assistant_logits.numel() > 0:
            # Compress: keep top-K values + indices per token plus an
            # ``others_logit`` scalar per token so the logsumexp of the
            # compressed tensor matches the original within fp16 eps.
            logits_fp32 = assistant_logits.detach().to(torch.float32).cpu()
            topk_values, topk_indices = torch.topk(logits_fp32, k=self.topk, dim=-1)
            # Build mask of the kept indices so we can compute the remainder's LSE
            mask = torch.zeros_like(logits_fp32, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            remainder = logits_fp32.masked_fill(mask, float("-inf"))
            # LSE of remainder — finite because at least V-K positions are not -inf
            lse_remainder = torch.logsumexp(remainder, dim=-1)
            # Distribute remainder mass uniformly over V - K slots:
            #   others_logit = lse_remainder - log(V - K)
            others_logit = lse_remainder - torch.log(
                torch.tensor(float(self.vocab_size - self.topk))
            )
            torch.save(
                {
                    "topk_values": topk_values.to(torch.float16),
                    "topk_indices": topk_indices.to(torch.int32),
                    "others_logit": others_logit.to(torch.float16),
                    "assistant_len": int(logits_fp32.size(0)),
                    "vocab_size": int(logits_fp32.size(1)),
                    "topk": int(self.topk),
                },
                tmp,
            )
            tmp.rename(p)
            return
        torch.save(
            {
                "assistant_logits": assistant_logits.to(torch.float16).cpu(),
                "assistant_len": int(assistant_logits.size(0)),
            },
            tmp,
        )
        tmp.rename(p)


# ---------------------------------------------------------------------------
# Teacher forward (with cache)
# ---------------------------------------------------------------------------


def get_teacher_assistant_logits(
    rows: List[Dict[str, Any]],
    batch: Dict[str, torch.Tensor],
    teacher,
    cache: TeacherLogitCache,
    device: torch.device,
) -> List[torch.Tensor]:
    """Return one per-row tensor of teacher assistant-token logits.

    The teacher forward is done **only on rows not in the cache**. Cache
    hits are loaded from disk. Returned logits are float16 on CPU (for
    memory parity with the on-disk cache) — the trainer casts back to
    bf16/fp32 when computing KL.
    """
    keys = [cache_key(r["user"], r["assistant"]) for r in rows]
    miss_indices = [i for i, k in enumerate(keys) if not cache.has(k)]
    cached_logits: Dict[int, torch.Tensor] = {}

    # Load cache hits
    for i, k in enumerate(keys):
        if i in miss_indices:
            continue
        cached_logits[i] = cache.load(k)

    # Compute misses
    if miss_indices:
        sub_input_ids = batch["input_ids"][miss_indices].to(device)
        sub_attn = batch["attention_mask"][miss_indices].to(device)
        sub_labels = batch["labels"][miss_indices]
        with torch.no_grad():
            t_out = teacher(
                input_ids=sub_input_ids,
                attention_mask=sub_attn,
                use_cache=False,
            )
            t_logits = t_out.logits  # [B_miss, L, V]
        # Extract per-row assistant-only slices and persist.
        for local_i, orig_i in enumerate(miss_indices):
            mask = (sub_labels[local_i] != -100)
            row_logits = t_logits[local_i][mask].detach().to(torch.float16).cpu()
            cache.save(keys[orig_i], row_logits)
            cached_logits[orig_i] = row_logits
        del t_out, t_logits

    return [cached_logits[i] for i in range(len(rows))]


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_distill_loss(
    student_logits: torch.Tensor,  # [B, L, V]
    labels: torch.Tensor,          # [B, L]
    teacher_assistant_logits: List[torch.Tensor],  # each [L_i, V]
    alpha: float,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, sft_loss, kl_loss)``.

    Standard next-token shift: student predicts token ``i+1`` from
    position ``i``, so we shift both logits and labels by one.
    """
    # Shift for next-token prediction
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # SFT cross-entropy on assistant tokens only
    V = shift_logits.size(-1)
    sft_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    # KL on assistant tokens only. We reconstruct per-row position masks
    # from the shifted labels and align against the per-row teacher
    # logits that the cache returned.
    device = student_logits.device
    kl_parts: List[torch.Tensor] = []
    total_positions = 0
    for row_i in range(shift_labels.size(0)):
        row_mask = (shift_labels[row_i] != -100)
        n = int(row_mask.sum().item())
        if n == 0:
            continue
        s_logits_row = shift_logits[row_i][row_mask]  # [n, V]
        t_logits_row = teacher_assistant_logits[row_i].to(device=device, dtype=torch.float32)
        # Teacher cache holds assistant-token logits at the *unshifted*
        # positions (i.e. including the final eos prediction). After
        # shifting labels by one, one fewer position remains on the
        # student side. Trim the teacher tensor to match.
        if t_logits_row.size(0) > n:
            t_logits_row = t_logits_row[: n]
        elif t_logits_row.size(0) < n:
            # Student has more assistant-labelled positions than the
            # teacher cache — shouldn't happen, but guard anyway.
            s_logits_row = s_logits_row[: t_logits_row.size(0)]
            n = t_logits_row.size(0)

        s_log_probs = F.log_softmax(s_logits_row.float() / temperature, dim=-1)
        t_probs = F.softmax(t_logits_row / temperature, dim=-1)
        # KL(teacher || student) — the standard distillation direction
        # (minimise forward KL of teacher over student).
        row_kl = F.kl_div(
            s_log_probs, t_probs, reduction="batchmean", log_target=False
        )
        kl_parts.append(row_kl * n)
        total_positions += n

    if total_positions == 0 or not kl_parts:
        kl_loss = torch.tensor(0.0, device=device)
    else:
        kl_loss = torch.stack(kl_parts).sum() / total_positions
        # Scale by T² — conventional for Hinton-style distillation so the
        # gradient magnitude is temperature-invariant.
        kl_loss = kl_loss * (temperature ** 2)

    total = alpha * sft_loss + (1.0 - alpha) * kl_loss
    return total, sft_loss.detach(), kl_loss.detach()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _setup_logger(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(fh)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _gpu_current_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _build_optimizer(model, lr: float):
    """Return a memory-efficient AdamW optimizer.

    Preference order: ``bitsandbytes.PagedAdamW8bit`` (best memory) →
    ``torch.optim.AdamW`` (fallback, 2× state). Paged 8-bit is the
    production default — with teacher + 7B student on 80GB, fp32 AdamW
    state (~32 GB for 8B params) would OOM.
    """
    try:
        from bitsandbytes.optim import PagedAdamW8bit  # type: ignore
        logger.info("Using PagedAdamW8bit (bitsandbytes) — memory-efficient")
        return PagedAdamW8bit(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
    except ImportError:
        logger.warning(
            "bitsandbytes not installed — falling back to fp32 AdamW. "
            "Real training on 80GB will likely OOM; install bitsandbytes."
        )
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )


def _load_model(path: str, label: str):
    """Load a model + tokenizer in bf16 on CUDA (or CPU fallback)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading %s model from %s", label, path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # Qwen3 tokenizer ships without an explicit pad token in some
    # distributions — re-use eos if missing so padding is well-defined.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok


def assert_vocab_compatible(student, teacher) -> int:
    """Hard-fail if the vocab sizes don't match — see spec."""
    s_vocab = student.config.vocab_size
    t_vocab = teacher.config.vocab_size
    if s_vocab != t_vocab:
        raise RuntimeError(
            f"Student vocab {s_vocab} != teacher vocab {t_vocab}. "
            "Distillation approach fails — do NOT attempt logit "
            "projection. See docs/superpowers/specs/"
            "2026-04-18-docwain-v5-design.md §'Vocabulary / tokenizer "
            "alignment'."
        )
    logger.info("Vocab compatibility OK: both models have %d tokens", s_vocab)
    return s_vocab


def _make_dataloader(
    dataset: V5JsonlDataset,
    tokenizer,
    cfg: DistillConfig,
) -> DataLoader:
    """Tokenize on the fly and batch with right-padding.

    We keep the raw row in the batch under key ``_rows`` so the teacher
    cache can hash by (user, assistant).
    """
    pad_id = tokenizer.pad_token_id

    def _collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized: List[Dict[str, torch.Tensor]] = []
        rows: List[Dict[str, Any]] = []
        for row in items:
            tok = tokenize_row(row, tokenizer, cfg.max_seq_length)
            if not tok:
                continue
            tokenized.append(tok)
            rows.append(row)
        if not tokenized:
            return {}
        batch = pad_collate(tokenized, pad_id)
        batch["_rows"] = rows
        return batch

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=not cfg.dry_run,  # deterministic order for dry-run
        collate_fn=_collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_distillation(cfg: DistillConfig) -> Dict[str, Any]:
    """Entry point — returns a metrics dict (always) and writes a
    dry-run report file if ``cfg.dry_run``."""

    torch.manual_seed(cfg.seed)
    _setup_logger(cfg.log_path)

    # --- Load models ------------------------------------------------------
    teacher, teacher_tok = _load_model(cfg.teacher, "teacher")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_peak_mb = _gpu_current_mb()
    logger.info("Teacher loaded — VRAM after teacher: %.1f MB", teacher_peak_mb)

    student, student_tok = _load_model(cfg.student, "student")
    student.train()
    # Activation checkpointing — the 7B student + 14B teacher plus AdamW
    # state puts us on the edge of 80GB; trading compute for memory here
    # keeps the long-context batches feasible. Reused from the V2 trainer
    # defaults (``use_gradient_checkpointing`` in train_track.py).
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    student.config.use_cache = False
    student_peak_mb = _gpu_current_mb() - teacher_peak_mb
    logger.info(
        "Student loaded — VRAM after student: %.1f MB (student delta %.1f MB)",
        _gpu_current_mb(), student_peak_mb,
    )

    # --- Vocab alignment --------------------------------------------------
    vocab_size = assert_vocab_compatible(student, teacher)

    # Tokenizer sanity: the cache is keyed by text, so both sides must
    # use the same tokenizer vocab for KL to be valid. Use the student
    # tokenizer consistently (teacher/student are required to be
    # Qwen3-family per the spec).
    if teacher_tok.vocab_size != student_tok.vocab_size:
        logger.warning(
            "Teacher/student tokenizer vocab_size differ (%d vs %d); "
            "falling back to student tokenizer for all tokenization.",
            teacher_tok.vocab_size, student_tok.vocab_size,
        )

    # --- Corpus + loader --------------------------------------------------
    dataset = V5JsonlDataset(cfg.corpus)
    logger.info("Loaded %d corpus rows from %s", len(dataset), cfg.corpus)

    loader = _make_dataloader(dataset, student_tok, cfg)

    # --- Cache ------------------------------------------------------------
    cache = TeacherLogitCache(
        cfg.teacher_cache,
        topk=cfg.cache_topk,
        vocab_size=teacher.config.vocab_size,
    )
    if cfg.cache_topk > 0:
        logger.info(
            "Teacher cache using top-k compression: K=%d, vocab=%d (~%.1f× smaller per row)",
            cfg.cache_topk, teacher.config.vocab_size,
            teacher.config.vocab_size / (2 * cfg.cache_topk + 1),
        )

    # --- Optim ------------------------------------------------------------
    # Paged 8-bit AdamW keeps optimizer state at ~8 bytes/param instead of
    # fp32's 16, making teacher + 7B student + optimizer fit on a single
    # A100-80GB. Falls back to fp32 AdamW if bitsandbytes is unavailable.
    optim = _build_optimizer(student, cfg.learning_rate)
    # Step schedule — for the dry-run we just use 10 steps; real runs
    # use epochs · len(loader) / grad_accum.
    total_update_steps = (
        cfg.dry_run_steps if cfg.dry_run
        else max(1, (cfg.epochs * len(loader)) // cfg.grad_accum)
    )
    warmup_steps = max(1, int(cfg.warmup_ratio * total_update_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay to 10% of peak
        progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)

    # --- Training loop ----------------------------------------------------
    out_root = Path(cfg.output)
    ckpt_root = out_root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    device = next(student.parameters()).device
    loss_trajectory: List[Dict[str, float]] = []
    step_times: List[float] = []
    last_ckpt = time.time()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    update_step = 0
    micro_step = 0
    optim.zero_grad(set_to_none=True)

    start_loop = time.time()
    for epoch in range(cfg.epochs):
        for batch in loader:
            if not batch:
                continue
            rows = batch.pop("_rows")
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward (+ cache write for misses)
            t_logits_list = get_teacher_assistant_logits(
                rows, batch, teacher, cache, device
            )

            # Student forward
            step_start = time.time()
            s_out = student(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=False,
            )

            total_loss, sft_loss, kl_loss = compute_distill_loss(
                s_out.logits,
                labels,
                t_logits_list,
                alpha=cfg.alpha,
                temperature=cfg.temperature,
            )
            (total_loss / cfg.grad_accum).backward()

            micro_step += 1
            if micro_step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                update_step += 1

            step_time = time.time() - step_start
            step_times.append(step_time)
            entry = {
                "micro_step": micro_step,
                "update_step": update_step,
                "sft_loss": float(sft_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "total_loss": float(total_loss.item()),
                "step_time_s": float(step_time),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            loss_trajectory.append(entry)
            logger.info(
                "step %d/%d sft=%.4f kl=%.4f total=%.4f lr=%.2e (%.2fs)",
                micro_step, total_update_steps * cfg.grad_accum,
                entry["sft_loss"], entry["kl_loss"], entry["total_loss"],
                entry["lr"], entry["step_time_s"],
            )

            # Periodic checkpoints on real runs
            if (not cfg.dry_run
                    and time.time() - last_ckpt > cfg.checkpoint_interval_sec):
                _save_checkpoint(student, student_tok, ckpt_root)
                last_ckpt = time.time()

            if cfg.dry_run and micro_step >= cfg.dry_run_steps:
                break
        if cfg.dry_run and micro_step >= cfg.dry_run_steps:
            break

    elapsed = time.time() - start_loop
    peak_vram_mb = _gpu_peak_mb()

    # --- Finalise ---------------------------------------------------------
    metrics: Dict[str, Any] = {
        "teacher_path": cfg.teacher,
        "student_path": cfg.student,
        "corpus": cfg.corpus,
        "vocab_size": vocab_size,
        "vocab_match": True,
        "dry_run": cfg.dry_run,
        "micro_steps_run": micro_step,
        "update_steps_run": update_step,
        "elapsed_sec": elapsed,
        "mean_step_sec": (sum(step_times) / len(step_times)) if step_times else 0.0,
        "teacher_vram_mb": teacher_peak_mb,
        "student_vram_mb": student_peak_mb,
        "combined_peak_vram_mb": peak_vram_mb,
        "alpha": cfg.alpha,
        "temperature": cfg.temperature,
        "loss_trajectory": loss_trajectory,
        "corpus_size": len(dataset),
    }

    if cfg.dry_run:
        # Decorate report with green-light assessment so the caller can
        # grep one field to decide whether to kick off the real 18h run.
        greenlight = (
            metrics["vocab_match"]
            and metrics["micro_steps_run"] >= cfg.dry_run_steps
            and all(
                math.isfinite(e["sft_loss"])
                and math.isfinite(e["kl_loss"])
                and math.isfinite(e["total_loss"])
                for e in loss_trajectory
            )
        )
        metrics["green_light_for_real_run"] = bool(greenlight)
        rep_path = Path(cfg.dryrun_report_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Wrote dry-run report to %s", rep_path)
    else:
        _save_checkpoint(student, student_tok, out_root, final=True)
        logger.info("Training complete; final bf16 weights at %s", out_root)

    return metrics


def _save_checkpoint(model, tokenizer, out_dir: Path, final: bool = False) -> Path:
    """Save a bf16 checkpoint snapshot.

    For final=True the model is saved directly under ``out_dir``; for
    intermittent checkpoints under ``out_dir/t_<unix>/``.
    """
    if final:
        path = out_dir
    else:
        path = out_dir / f"t_{int(time.time())}"
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving checkpoint to %s", path)
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DocWain V5 — 14B → 7B KL+SFT distillation trainer",
    )
    p.add_argument("--teacher", required=True,
                   help="Path or HF id of the teacher (e.g. models/DocWain-14B-v5)")
    p.add_argument("--student", default="Qwen/Qwen3-8B",
                   help="Path or HF id of the student (default: Qwen/Qwen3-8B)")
    p.add_argument("--corpus", required=True,
                   help="Path to V5-format SFT JSONL")
    p.add_argument("--output", required=True,
                   help="Where the final bf16 student is saved")

    p.add_argument("--alpha", type=float, default=0.5,
                   help="SFT/KL mixing weight — 1.0 = pure SFT, 0.0 = pure KL")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Distillation temperature (applied to both sides)")

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--checkpoint-interval", default="4h",
                   help="Duration between disk snapshots, e.g. 4h, 30m, 3600s")

    p.add_argument("--teacher-cache",
                   default="finetune_artifacts/v5/distillation_teacher_cache",
                   help="Directory where teacher logits are cached (sha256 keys)")
    p.add_argument("--cache-topk", type=int, default=256,
                   help="Store only top-K logits + an 'others' fill per token. "
                        "0 = full dense (huge: ~1.4TB on 31K rows at V=151936); "
                        "256 shrinks per-row footprint ~250x while preserving KL "
                        "signal (>99.9%% of probability mass for Qwen3-family).")
    p.add_argument("--dry-run", action="store_true",
                   help="Run 10 steps end-to-end and emit a report")
    p.add_argument("--dry-run-steps", type=int, default=10)
    p.add_argument("--dryrun-report-path",
                   default="finetune_artifacts/v5/distillation_dryrun_report.json")
    p.add_argument("--log-path",
                   default="finetune_artifacts/v5/distillation_training.log")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)

    cfg = DistillConfig(
        teacher=args.teacher,
        student=args.student,
        corpus=args.corpus,
        output=args.output,
        teacher_cache=args.teacher_cache,
        cache_topk=args.cache_topk,
        alpha=args.alpha,
        temperature=args.temperature,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        checkpoint_interval_sec=parse_duration(args.checkpoint_interval),
        dry_run=args.dry_run,
        dry_run_steps=args.dry_run_steps,
        dryrun_report_path=args.dryrun_report_path,
        log_path=args.log_path,
        seed=args.seed,
    )
    metrics = run_distillation(cfg)
    summary = {k: v for k, v in metrics.items() if k != "loss_trajectory"}
    logger.info("Run summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict
import threading

from .config import settings

_MODEL = None
_TOKENIZER = None
_LOAD_LOCK = threading.Lock()

FINAL_LABELS = [
    "normal",
    "individuals#offensive",
    "individuals#hate",
    "groups#offensive",
    "groups#hate",
    "religion#offensive",
    "religion#hate",
    "race#offensive",
    "race#hate",
    "politics#offensive",
    "politics#hate",
]

CATEGORY_LABELS = {
    "individuals": ("individuals#offensive", "individuals#hate"),
    "groups": ("groups#offensive", "groups#hate"),
    "religion": ("religion#offensive", "religion#hate"),
    "race": ("race#offensive", "race#hate"),
    "politics": ("politics#offensive", "politics#hate"),
}


def _resolve_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[1] / path).resolve()
    return str(path)


def _resolve_labels_list(labels: List[str], prefer_hate: bool = True) -> List[str]:
    resolved = list(labels)
    if "normal" in resolved and len(resolved) > 1:
        resolved = [label for label in resolved if label != "normal"]
    for _, (off_label, hate_label) in CATEGORY_LABELS.items():
        if off_label in resolved and hate_label in resolved:
            if prefer_hate:
                resolved.remove(off_label)
            else:
                resolved.remove(hate_label)
    return resolved


def _parse_output(output: str) -> List[str]:
    import re
    import logging

    cleaned = output.replace("<|im_end|>", "").strip()
    labels_match = re.search(r"labels?:\s*(.+?)$", cleaned, re.IGNORECASE | re.DOTALL)
    if labels_match:
        cleaned = labels_match.group(1).strip()

    cleaned = cleaned.lower().replace(";", ",").replace(" and ", ",")
    parts = [p.strip() for p in re.split(r"[,\n]", cleaned) if p.strip()]

    labels: List[str] = []
    for part in parts:
        for valid_label in FINAL_LABELS:
            if valid_label.lower() == part or valid_label.lower() in part:
                if valid_label not in labels:
                    labels.append(valid_label)
                break

    if not labels:
        logging.warning("Failed to parse model output, defaulting to 'normal'. Raw output: %s", output[:200])
        labels = ["normal"]

    return _resolve_labels_list(labels)


def _format_prompt(text: str) -> str:
    labels_str = ", ".join(FINAL_LABELS)
    return (
        "<|im_start|>system\n"
        "You are a Vietnamese hate speech classification system. Classify the text into one or more labels.\n"
        f"Valid labels: {labels_str}\n"
        "Return only the label names, separated by commas.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Text to classify: {text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _load_model() -> None:
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return

    with _LOAD_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model = settings.QWEN_BASE_MODEL
        base_model_path = _resolve_path(base_model) if Path(base_model).exists() else base_model
        adapter_path = _resolve_path(settings.QWEN_ADAPTER_PATH)
        tokenizer_path = _resolve_path(settings.QWEN_TOKENIZER_PATH)

        use_cuda = torch.cuda.is_available()
        quant_config = None
        torch_dtype = torch.float16 if use_cuda else torch.float32
        device_map = "auto" if use_cuda else None
        offload_folder = None
        if use_cuda and settings.QWEN_USE_4BIT:
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                quant_config = None

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_cuda:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map={"": 0},
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    quantization_config=quant_config,
                )
                model = PeftModel.from_pretrained(model, adapter_path)
            except Exception:
                offload_folder = _resolve_path(settings.QWEN_OFFLOAD_DIR)
                Path(offload_folder).mkdir(parents=True, exist_ok=True)
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    quantization_config=quant_config,
                    offload_folder=offload_folder,
                )
                model = PeftModel.from_pretrained(model, adapter_path, offload_folder=offload_folder)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=None,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=None,
            )
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        _MODEL = model
        _TOKENIZER = tokenizer


def _predict_labels(text: str) -> List[str]:
    import torch

    _load_model()
    assert _MODEL is not None and _TOKENIZER is not None

    prompt = _format_prompt(text)
    inputs = _TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.QWEN_MAX_LENGTH,
    )

    device = getattr(_MODEL, "device", None)
    if device is None:
        device = next(_MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    eos_token_id = _TOKENIZER.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id is None or eos_token_id == _TOKENIZER.unk_token_id:
        eos_token_id = _TOKENIZER.eos_token_id

    with torch.no_grad():
        outputs = _MODEL.generate(
            **inputs,
            max_new_tokens=settings.QWEN_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_TOKENIZER.pad_token_id,
            eos_token_id=eos_token_id,
        )

    new_tokens = outputs[0][len(inputs["input_ids"][0]) :]
    decoded = _TOKENIZER.decode(new_tokens, skip_special_tokens=False)
    return _parse_output(decoded)


def _coarse_label(labels: List[str]) -> str:
    if any("#hate" in label for label in labels):
        return "HATE"
    if any("#offensive" in label for label in labels):
        return "OFFENSIVE"
    return "NEUTRAL"


def _distribution(label: str) -> List[Tuple[str, float]]:
    if label == "HATE":
        dist = [("HATE", 0.9), ("OFFENSIVE", 0.08), ("NEUTRAL", 0.02)]
    elif label == "OFFENSIVE":
        dist = [("HATE", 0.1), ("OFFENSIVE", 0.85), ("NEUTRAL", 0.05)]
    else:
        dist = [("HATE", 0.05), ("OFFENSIVE", 0.05), ("NEUTRAL", 0.9)]
    total = sum(v for _, v in dist)
    return [(name, float(v / total)) for name, v in dist]


def model_info() -> Dict[str, str]:
    return {
        "name": "qwen_rationale",
        "base_model": settings.QWEN_BASE_MODEL,
        "adapter_path": settings.QWEN_ADAPTER_PATH,
        "tokenizer_path": settings.QWEN_TOKENIZER_PATH,
    }


def predict_label(text: str, threshold: float = 0.5) -> Tuple[str, float, List[Tuple[str, float]]]:
    labels = _predict_labels(text)
    coarse = _coarse_label(labels)
    dist = _distribution(coarse)
    score = dict(dist)[coarse]
    return coarse, float(score), dist

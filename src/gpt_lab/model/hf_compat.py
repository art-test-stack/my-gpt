"""Hugging Face configuration compatibility for native, from-scratch models.

Only configuration JSON is read here.  This module deliberately has no model
loading APIs: gpt-lab always creates fresh ``DenseTransformer`` weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from gpt_lab.utils.schemas import CompatibilityReport, TransformerConfig


def _value(raw: Mapping[str, Any], *names: str, default: Any = None) -> tuple[str, Any]:
    for name in names:
        if name in raw:
            return name, raw[name]
    return names[0], default


def _gpt2(raw: Mapping[str, Any], report: CompatibilityReport) -> TransformerConfig:
    required = {
        "vocab_size": ("vocab_size",), "max_context": ("n_positions", "max_position_embeddings"),
        "d_model": ("n_embd", "hidden_size"), "n_layers": ("n_layer", "num_hidden_layers"),
        "n_heads": ("n_head", "num_attention_heads"),
    }
    values: dict[str, Any] = {}
    for target, names in required.items():
        source, value = _value(raw, *names)
        if value is None:
            raise ValueError(f"GPT-2 configuration is missing required field {names[0]!r}")
        values[target] = value
        report.add("mapped", source, value, reason=f"mapped to {target}")
    if values["d_model"] % values["n_heads"]:
        raise ValueError("GPT-2 n_embd/hidden_size must be divisible by n_head/num_attention_heads")
    inner_name, inner = _value(raw, "n_inner", "intermediate_size")
    if inner is None:
        inner = 4 * values["d_model"]
        report.add("derived", inner_name, inner, reason="None derives GPT-2's 4 * hidden size MLP width")
    else:
        report.add("mapped", inner_name, inner, reason="mapped to d_ffn")
    tie_name, tie = _value(raw, "tie_word_embeddings", default=True)
    report.add("mapped", tie_name, tie, reason="mapped to tied embeddings")
    d_head = values["d_model"] // values["n_heads"]
    report.add("derived", "d_head", d_head, reason="hidden size / attention heads")
    report.add("derived", "n_kv_heads", values["n_heads"], reason="GPT-2 uses multi-head attention")
    config = TransformerConfig(
        vocab_size=values["vocab_size"], max_context=values["max_context"],
        d_model=values["d_model"], d_ffn=inner, n_layers=values["n_layers"],
        n_heads=values["n_heads"], n_kv_heads=values["n_heads"], d_head=d_head,
        tie_word_embeddings=bool(tie), window_pattern="L",
    )
    todo_fields = {
        "n_positions": ("transformer.position_encoding", "learned absolute positional embeddings are not represented by gpt-lab RoPE"),
        "activation_function": ("transformer.mlp", "GPT-2 GELU activation differs from gpt-lab SwiGLU"),
        "resid_pdrop": ("transformer.attention", "residual dropout is not represented"),
        "embd_pdrop": ("transformer.position_encoding", "embedding dropout is not represented"),
        "attn_pdrop": ("transformer.attention", "attention dropout is not represented"),
        "scale_attn_weights": ("transformer.attention", "GPT-2 attention scaling variant is not configurable"),
        "scale_attn_by_inverse_layer_idx": ("transformer.attention", "layer-index attention scaling is not configurable"),
        "reorder_and_upcast_attn": ("transformer.attention", "attention upcast/reordering is not configurable"),
        "initializer_range": ("transformer.initialization", "GPT-2 initializer range is not configurable"),
    }
    for name, (component, reason) in todo_fields.items():
        if name in raw and (raw[name] not in (False, 0, 0.0, None)):
            report.add("todos", name, raw[name], component=component, severity="architecture", reason=reason)
    report.add("todos", "layer_norm_epsilon", raw.get("layer_norm_epsilon", 1e-5), component="transformer.normalization", severity="architecture", reason="GPT-2 LayerNorm differs from gpt-lab RMSNorm")
    report.add("todos", "bias", True, component="transformer.attention", severity="architecture", reason="GPT-2 attention/MLP/projection biases are not configurable")
    ignored = {"model_type", "architectures", "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "use_cache", "return_dict", "output_attentions", "output_hidden_states", "summary_type", "summary_use_proj", "summary_activation", "summary_proj_to_labels", "summary_first_dropout", "task_specific_params", "id2label", "label2id", "transformers_version"}
    classified = set(todo_fields) | {key for names in required.values() for key in names} | {"n_inner", "intermediate_size", "tie_word_embeddings", "layer_norm_epsilon", "bias"}
    for name, value in raw.items():
        if name in ignored or name not in classified:
            report.add("ignored", name, value, reason="metadata, token/generation control, or not relevant to from-scratch architecture")
    report.finalise(config)
    return config


ADAPTERS: dict[str, Callable[[Mapping[str, Any], CompatibilityReport], TransformerConfig]] = {"gpt2": _gpt2}


def map_hf_config(raw: Mapping[str, Any], *, source: str = "<memory>", requested_revision: str | None = None, resolved_revision: str | None = None) -> tuple[TransformerConfig, CompatibilityReport]:
    model_type = raw.get("model_type")
    if not isinstance(model_type, str) or model_type not in ADAPTERS:
        raise ValueError(f"No gpt-lab compatibility adapter for Hugging Face model_type {model_type!r}")
    report = CompatibilityReport(
        source=source,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        model_type=model_type,
        adapter=f"{model_type} adapter",
    )
    return ADAPTERS[model_type](raw, report), report


def load_hf_config(source: str, *, revision: str | None = None, local_files_only: bool = False) -> tuple[dict[str, Any], str | None]:
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if path.is_file():
        return json.loads(path.read_text()), None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to resolve a remote --hf-config repository") from exc
    config_path = hf_hub_download(repo_id=source, filename="config.json", revision=revision, local_files_only=local_files_only)
    return json.loads(Path(config_path).read_text()), revision


def print_report(report: CompatibilityReport, parameter_count: int) -> str:
    return (f"HF compatibility: {report.status} ({report.adapter}; {report.model_type})\n"
            f"mapped={len(report.mapped)}, derived={len(report.derived)}, todos={len(report.todos)}, ignored={len(report.ignored)}\n"
            f"native parameter count: {parameter_count:,}")

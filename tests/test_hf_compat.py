import json

import pytest
from pydantic import BaseModel

from gpt_lab.model.hf_compat import load_hf_config, map_hf_config
from gpt_lab.utils.schemas import CompatibilityItem, CompatibilityReport


def tiny_gpt2(**updates):
    config = dict(model_type="gpt2", vocab_size=257, n_positions=64, n_embd=32,
                  n_layer=2, n_head=4, n_inner=128, tie_word_embeddings=True,
                  activation_function="gelu_new", resid_pdrop=.1, embd_pdrop=.1,
                  attn_pdrop=.1, layer_norm_epsilon=1e-5, bos_token_id=1,
                  eos_token_id=2, use_cache=True)
    config.update(updates)
    return config


def test_gpt2_mapping_and_report_from_local_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_gpt2()))
    raw, resolved = load_hf_config(str(path), local_files_only=True)
    config, report = map_hf_config(raw, source=str(path), resolved_revision=resolved)
    assert (config.vocab_size, config.max_context, config.d_model, config.d_ffn, config.n_layers) == (257, 64, 32, 128, 2)
    assert (config.n_heads, config.n_kv_heads, config.d_head) == (4, 4, 8)
    assert report.status == "partial"
    assert any(item.field == "activation_function" for item in report.todos)
    assert any(item.field == "bos_token_id" for item in report.ignored)


def test_none_n_inner_is_derived():
    config, report = map_hf_config(tiny_gpt2(n_inner=None))
    assert config.d_ffn == 128
    assert any(item.field == "n_inner" and item.value == 128 for item in report.derived)


def test_invalid_gpt2_dimensions_fail_usefully():
    with pytest.raises(ValueError, match="divisible"):
        map_hf_config(tiny_gpt2(n_embd=30, n_head=4))


def test_compatibility_payload_is_structured():
    _, report = map_hf_config(tiny_gpt2())
    assert isinstance(report, CompatibilityReport)
    assert isinstance(report.mapped[0], CompatibilityItem)
    assert isinstance(report, BaseModel)
    payload = report.as_dict()
    assert payload["resolved_gpt_lab_model_config"]["d_head"] == 8
    assert {"mapped", "derived", "ignored", "todos", "warnings"} <= payload.keys()

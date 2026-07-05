from __future__ import annotations

try:
    from transformers.models.smollm3.modeling_smollm3 import (
        SmolLM3Attention,
        apply_rotary_pos_emb as smollm3_apply_rotary_pos_emb,
        eager_attention_forward as smollm3_eager_attention_forward,
    )
    HAS_SMOLLM3 = True
except (ImportError, ModuleNotFoundError):
    SmolLM3Attention = None
    smollm3_apply_rotary_pos_emb = None
    smollm3_eager_attention_forward = None
    HAS_SMOLLM3 = False

def ensure_attention_compat_views(module):
    return module

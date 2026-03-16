"""Self-speculative decoding: use the same model with layer skipping as draft.

No separate draft model needed. No tokenizer matching. Works with any model.
The draft pass runs a subset of layers (e.g., every 10th) for fast predictions.
The verify pass runs ALL layers and accepts/rejects draft tokens.

Quality is IDENTICAL to normal decoding — the full model verifies everything.

Usage:
  export EXO_SPECULATIVE_DECODE=10  # skip factor (use every 10th layer)
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Any, List, Optional

from .models.base import create_attention_mask


class LayerSkipDraftModel(nn.Module):
    """Wraps an existing model, running only a subset of layers for fast drafting.

    Uses the SAME weights as the main model — no additional memory.
    Creates its own KV cache entries only for the selected layers.
    """

    def __init__(self, model: nn.Module, skip_factor: int = 10):
        super().__init__()
        self._full_model = model
        self.skip_factor = skip_factor

        # Get the inner model (handle Model → Qwen3MoeModel wrapping)
        inner = model.model if hasattr(model, "model") else model
        self._inner = inner
        all_layers = inner.layers

        # Select layers: every Nth, always include first and last
        indices = list(range(0, len(all_layers), skip_factor))
        if indices[-1] != len(all_layers) - 1:
            indices.append(len(all_layers) - 1)
        self._layer_indices = indices
        self._selected_layers = [all_layers[i] for i in indices]

        # Copy model args for compatibility
        if hasattr(model, "args"):
            self.args = model.args

    @property
    def layers(self) -> List[Any]:
        """Return selected layers — used by make_prompt_cache to size the cache."""
        return self._selected_layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        # Embedding
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self._inner.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self._selected_layers)

        mask = create_attention_mask(h, cache[0])

        # Run only selected layers
        for layer, c in zip(self._selected_layers, cache):
            h = layer(h, mask, c)

        # Final norm
        h = self._inner.norm(h)

        # LM head
        if hasattr(self._full_model, "args") and self._full_model.args.tie_word_embeddings:
            return self._inner.embed_tokens.as_linear(h)
        elif hasattr(self._full_model, "lm_head"):
            return self._full_model.lm_head(h)
        else:
            return self._inner.embed_tokens.as_linear(h)

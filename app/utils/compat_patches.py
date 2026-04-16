"""
Compatibility patches for third-party HuggingFace models loaded via
trust_remote_code=True whose remote code targets an older transformers API.

Both patches are idempotent — safe to call multiple times.

Apply once at process start (or before the first model load) by calling:

    from app.utils.compat_patches import apply_all_patches
    apply_all_patches()
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Patch 1 — XLMRoberta tokenizer ──────────────────────────────────────────
#
# transformers 5.x: XLMRobertaTokenizerFast no longer exposes
# build_inputs_with_special_tokens as a Python method (delegated to Rust).
# The Zilliz semantic-highlight model's remote code calls it directly.
#
def patch_xlm_roberta_tokenizer() -> None:
    try:
        from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast

        def _build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
        ) -> List[int]:
            bos = [self.bos_token_id]
            eos = [self.eos_token_id]
            if token_ids_1 is None:
                return bos + list(token_ids_0) + eos
            return bos + list(token_ids_0) + eos + eos + list(token_ids_1) + eos

        for cls in (XLMRobertaTokenizer, XLMRobertaTokenizerFast):
            if not hasattr(cls, "build_inputs_with_special_tokens"):
                cls.build_inputs_with_special_tokens = _build_inputs_with_special_tokens  # type: ignore[attr-defined]
                logger.debug("compat_patches: added build_inputs_with_special_tokens to %s", cls.__name__)

    except ImportError:
        pass


# ── Patch 2 — all_tied_weights_keys ─────────────────────────────────────────
#
# transformers 5.x: PreTrainedModel._finalize_model_loading calls
# model.all_tied_weights_keys (a new computed property). Remote model code
# written for transformers 4.x defines the older _tied_weights_keys attribute
# instead. When the property getter raises AttributeError internally,
# Python falls through to torch.nn.Module.__getattr__ which raises a
# misleading "object has no attribute 'all_tied_weights_keys'" error.
#
# Fix: wrap mark_tied_weights_as_initialized so that any model lacking the
# new property gets it added (delegating to _tied_weights_keys) before the
# original method runs.
#
def patch_tied_weights_compat() -> None:
    try:
        from transformers import PreTrainedModel

        # Guard: only patch once
        if getattr(PreTrainedModel.mark_tied_weights_as_initialized, "_compat_patched", False):
            return

        _original = PreTrainedModel.mark_tied_weights_as_initialized

        def _patched_mark_tied(self, loading_info):  # type: ignore[override]
            try:
                val = self.all_tied_weights_keys
                # transformers 5.x expects a dict; older remote code returns a list.
                # Coerce a list to {key: key} so .keys() works downstream.
                if isinstance(val, list):
                    type(self).all_tied_weights_keys = property(
                        lambda s: {k: k for k in (getattr(s, "_tied_weights_keys", None) or [])}
                    )
            except AttributeError:
                # Remote model class only defines _tied_weights_keys (old API).
                # Inject all_tied_weights_keys as a dict property on its concrete class.
                type(self).all_tied_weights_keys = property(
                    lambda s: {k: k for k in (getattr(s, "_tied_weights_keys", None) or [])}
                )
                logger.debug(
                    "compat_patches: injected all_tied_weights_keys into %s",
                    type(self).__name__,
                )
            return _original(self, loading_info)

        _patched_mark_tied._compat_patched = True  # type: ignore[attr-defined]
        PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied

    except ImportError:
        pass


# ── Patch 3 — HHEM / T5 encoder embed_tokens weight tying ───────────────────
#
# transformers 5.x: for custom remote T5-based models (e.g. vectara/hhem),
# from_pretrained() does not re-invoke tie_weights() after loading, leaving
# encoder.embed_tokens.weight initialised to zeros even though shared.weight
# is correctly loaded.  The encoder then sees identical all-zero embeddings for
# every input and returns a constant score (~0.502) regardless of the text.
#
# Fix: call model.tie_weights() immediately after from_pretrained().
# This helper is called in HHEMValidator._load_model() on the loaded instance.
#
def fix_hhem_weight_tying(model: object) -> None:
    """Call tie_weights() on an HHEM model instance to restore encoder embeddings."""
    try:
        from torch.nn import Module as _Module
        if isinstance(model, _Module) and callable(getattr(model, "tie_weights", None)):
            model.tie_weights()  # type: ignore[union-attr]
            logger.debug("compat_patches: called tie_weights() on %s", type(model).__name__)
    except Exception:
        pass


def apply_all_patches() -> None:
    """Apply every compatibility patch. Call once before any model load."""
    patch_xlm_roberta_tokenizer()
    patch_tied_weights_compat()

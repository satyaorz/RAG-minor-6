from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Lightweight sentence_transformers stub
# ---------------------------------------------------------------------------
# sentence_transformers transitively imports torch, which takes 20-30 s the
# first time in a cold process.  We inject a minimal fake module so every
# test import is instant.  Individual tests that need real embeddings should
# remove this entry from sys.modules inside their setUp.
# ---------------------------------------------------------------------------
if "sentence_transformers" not in sys.modules:
    import numpy as _np  # noqa: E402 — numpy is always available

    _fake_st = types.ModuleType("sentence_transformers")

    class _FakeSentenceTransformer:  # noqa: N801
        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, texts, *, normalize_embeddings: bool = False, show_progress_bar: bool = False, **kw) -> _np.ndarray:
            return _np.zeros((len(texts), 384), dtype="float32")

    _fake_st.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = _fake_st


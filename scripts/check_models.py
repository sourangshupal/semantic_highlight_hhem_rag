"""
Model health check script.

Verifies that both ML models load correctly and produce sensible outputs
before running the notebook or the full API.

Usage:
    uv run python scripts/check_models.py
"""

import sys
import time
import traceback

# ── colour helpers (no deps) ─────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK   = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠  WARN{RESET}"


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


def check(label: str, ok: bool, detail: str = "") -> None:
    status = OK if ok else FAIL
    print(f"  {status}  {label}")
    if detail:
        indent = "        "
        for line in detail.splitlines():
            print(f"{indent}{line}")


# ── 1. Python & package versions ─────────────────────────────────────────────
section("1 / Environment")

import platform
print(f"  Python   : {sys.version.split()[0]}")
print(f"  Platform : {platform.platform()}")

try:
    import torch
    print(f"  torch    : {torch.__version__}")
    check("torch import", True)
except ImportError as e:
    check("torch import", False, str(e))
    sys.exit(1)

try:
    import transformers
    print(f"  transformers : {transformers.__version__}")
    check("transformers import", True)
except ImportError as e:
    check("transformers import", False, str(e))
    sys.exit(1)

try:
    import nltk
    print(f"  nltk     : {nltk.__version__}")
    check("nltk import", True)
except ImportError as e:
    check("nltk import", False, str(e))
    sys.exit(1)

# ── 2. NLTK data ──────────────────────────────────────────────────────────────
section("2 / NLTK data")

try:
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import PunktSentenceTokenizer  # noqa: F401
    check("punkt / punkt_tab downloaded", True)
except Exception as e:
    check("punkt / punkt_tab downloaded", False, str(e))

# ── 3. Compatibility patches ──────────────────────────────────────────────────
section("3 / Compatibility patches (transformers 5.x → remote model code)")

try:
    from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast, PreTrainedModel

    before_slow = hasattr(XLMRobertaTokenizer, "build_inputs_with_special_tokens")
    before_fast = hasattr(XLMRobertaTokenizerFast, "build_inputs_with_special_tokens")
    print(f"  XLMRobertaTokenizer     build_inputs before patch : {before_slow}")
    print(f"  XLMRobertaTokenizerFast build_inputs before patch : {before_fast}")

    sys.path.insert(0, ".")
    from app.utils.compat_patches import apply_all_patches
    apply_all_patches()

    after_slow = hasattr(XLMRobertaTokenizer, "build_inputs_with_special_tokens")
    after_fast = hasattr(XLMRobertaTokenizerFast, "build_inputs_with_special_tokens")
    tied_patched = getattr(
        PreTrainedModel.mark_tied_weights_as_initialized, "_compat_patched", False
    )

    check("XLMRobertaTokenizer.build_inputs_with_special_tokens present",     after_slow)
    check("XLMRobertaTokenizerFast.build_inputs_with_special_tokens present", after_fast)
    check("mark_tied_weights_as_initialized patched for all_tied_weights_keys compat", tied_patched)
except Exception as e:
    check("compat patches", False, traceback.format_exc())

# ── 4. SemanticHighlighter ────────────────────────────────────────────────────
section("4 / SemanticHighlighter  (zilliz/semantic-highlight-bilingual-v1)")

highlighter = None
t0 = time.time()
try:
    from app.services.semantic_highlighter import SemanticHighlighter
    highlighter = SemanticHighlighter()
    load_time = time.time() - t0
    check("model loaded", highlighter.is_loaded(), f"load time: {load_time:.1f}s")
except Exception as e:
    check("model loaded", False, traceback.format_exc())

if highlighter and highlighter.is_loaded():
    # Quick smoke test
    QUERY   = "What are the effects of climate change?"
    CONTEXT = (
        "Global temperatures have risen by 1.1°C since the pre-industrial era. "
        "Leonardo da Vinci painted the Mona Lisa around 1503. "
        "Sea levels have risen roughly 20 cm over the last century."
    )
    try:
        t0 = time.time()
        result = highlighter.highlight(query=QUERY, context=CONTEXT, threshold=0.5)
        inf_time = time.time() - t0

        kept      = result["highlighted_sentences"]
        scores    = result["sentence_probabilities"]
        comp_rate = result["compression_rate"]

        check("highlight() returns highlighted_sentences", isinstance(kept, list) and len(kept) > 0)
        check("highlight() returns sentence_probabilities", isinstance(scores, list) and len(scores) > 0)
        check("highlight() returns compression_rate",       isinstance(comp_rate, float))
        check(
            f"irrelevant sentence filtered (da Vinci score < 0.5)",
            all("da Vinci" not in s for s in kept),
            f"kept sentences: {kept}",
        )
        check(
            f"climate sentence retained",
            any("temperature" in s.lower() or "sea level" in s.lower() for s in kept),
            f"kept sentences: {kept}",
        )

        print(f"\n  {YELLOW}Sentence scores:{RESET}")
        for sent, score in zip(CONTEXT.split(". "), scores):
            bar = "█" * int(score * 20)
            print(f"    {score:.3f}  {bar:<20}  {sent[:60]}")
        print(f"  Compression rate : {comp_rate:.2%}  |  inference: {inf_time*1000:.0f}ms")

    except Exception as e:
        check("highlight() inference", False, traceback.format_exc())

# ── 5. HHEMValidator ─────────────────────────────────────────────────────────
section("5 / HHEMValidator  (vectara/hallucination_evaluation_model)")

validator = None
t0 = time.time()
try:
    from app.services.hhem_validator import HHEMValidator
    validator = HHEMValidator()
    load_time = time.time() - t0
    check("model loaded", validator.is_loaded(), f"load time: {load_time:.1f}s")
except Exception as e:
    check("model loaded", False, traceback.format_exc())

if validator and validator.is_loaded():
    CONTEXT = (
        "The Eiffel Tower is located in Paris, France. "
        "It was constructed in 1889 and stands 330 metres tall. "
        "It was designed by Gustave Eiffel."
    )
    # Faithful: every claim is verbatim from the context
    FAITHFUL_ANSWER = (
        "The Eiffel Tower is in Paris and was built in 1889. "
        "It stands 330 metres tall and was designed by Gustave Eiffel."
    )
    # Strongly hallucinated: wrong country, wrong year, wrong designer
    HALLUCINATED_ANSWER = (
        "The Eiffel Tower is located in Berlin, Germany. "
        "It was constructed in 1756 and was designed by Leonardo da Vinci."
    )

    try:
        t0 = time.time()
        faithful_result     = validator.validate(CONTEXT, FAITHFUL_ANSWER)
        hallucinated_result = validator.validate(CONTEXT, HALLUCINATED_ANSWER)
        inf_time = time.time() - t0

        f_score = faithful_result["score"]
        h_score = hallucinated_result["score"]

        check("validate() returns score for faithful answer",     isinstance(f_score, float))
        check("validate() returns score for hallucinated answer", isinstance(h_score, float))
        check(
            f"faithful answer scores higher than hallucinated  ({f_score:.3f} > {h_score:.3f})",
            f_score > h_score,
        )
        check(
            f"faithful answer NOT flagged  (score={f_score:.3f} >= 0.5)",
            not faithful_result["is_hallucinated"],
        )
        check(
            f"hallucinated answer IS flagged  (score={h_score:.3f} < 0.5)",
            hallucinated_result["is_hallucinated"],
        )

        print(f"\n  {YELLOW}HHEM scores:{RESET}")
        print(f"    Faithful answer     : {f_score:.3f}  {'✓ trusted' if not faithful_result['is_hallucinated'] else '⚠ flagged'}")
        print(f"    Hallucinated answer : {h_score:.3f}  {'✓ trusted' if not hallucinated_result['is_hallucinated'] else '⚠ flagged'}")
        print(f"  Inference time (both): {inf_time*1000:.0f}ms")

    except Exception as e:
        check("validate() inference", False, traceback.format_exc())

# ── 6. Summary ────────────────────────────────────────────────────────────────
section("6 / Summary")

all_ok = (
    highlighter is not None and highlighter.is_loaded()
    and validator  is not None and validator.is_loaded()
)

if all_ok:
    print(f"\n  {GREEN}{BOLD}Both models loaded and producing sensible outputs.{RESET}")
    print(f"  {GREEN}The notebook is ready to run.{RESET}\n")
    sys.exit(0)
else:
    print(f"\n  {RED}{BOLD}One or more models failed to load. Fix the errors above before running the notebook.{RESET}\n")
    sys.exit(1)

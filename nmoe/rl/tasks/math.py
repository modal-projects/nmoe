"""Math tasks for RLVR training.

Supports GSM8K and MATH datasets with verifiable numeric answers.
Includes sympy-based symbolic verification for algebraic equivalence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from nmoe.rl.rewards_harmony import CHANNELS, harmony_message, parse_harmony_text
from nmoe.rl.tasks import Task


# =============================================================================
# Sympy-based Symbolic Verification (from slime/deepscaler)
# =============================================================================

# Patterns that may cause sympy to hang
_BAD_SUBSTRINGS = ["^{", "^("]
_BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
_TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parse expression with sympy, handling ^ as power."""
    try:
        import sympy
        from sympy.parsing import sympy_parser
    except ImportError:
        return None

    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations +
            (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempt to parse LaTeX to sympy-readable expression."""
    try:
        from pylatexenc import latex2text
        expr = expr.replace("\\tfrac", "\\frac")
        expr = expr.replace("\\dfrac", "\\frac")
        expr = expr.replace("\\frac", " \\frac")  # Handle mixed numbers
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    except ImportError:
        # Fallback: basic LaTeX removal
        expr = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", expr)
        expr = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", expr)
        expr = expr.replace("\\pi", "pi")
        expr = expr.replace("\\infty", "inf")

    # Replace special chars
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _normalize_for_sympy(expr: str | None) -> str | None:
    """Heavy normalization for sympy comparison."""
    if expr is None:
        return None

    # Remove enclosing \text{}
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    # Remove symbols
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    # Handle large number words
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    # Remove units
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second",
        "minute", "hour", "day", "week", "month", "year",
        "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    # Remove enclosing braces
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)

    # Convert to int if it's a whole number
    try:
        val = float(expr)
        if abs(val - int(round(val))) <= 1e-7:
            expr = str(int(round(val)))
    except (ValueError, TypeError):
        pass

    # Parse LaTeX if present
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # Handle mixed numbers: "7 3/4" -> "7+3/4"
    expr = re.sub("- *", "-", expr)
    expr = re.sub(r"([0-9]) +([0-9])", r"\1+\2", expr)
    expr = expr.replace(" ", "")

    # Remove remaining LaTeX braces
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # Lowercase for text answers
    expr = expr.lower()

    # Strip commas from numbers
    expr = _strip_commas(expr)

    # Convert to int if whole number
    try:
        val = float(expr.replace(",", ""))
        if abs(val - int(round(val))) <= 1e-7:
            expr = str(int(round(val)))
    except (ValueError, TypeError):
        pass

    return expr


def _strip_commas(expr: str) -> str:
    """Strip properly formatted commas from numbers."""
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return expr


def _count_unknown_letters(expr: str) -> int:
    """Count unique letters that aren't part of known functions."""
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    expr = expr.replace("pi", "")
    expr = expr.replace("inf", "")
    expr = expr.replace("sin", "")
    expr = expr.replace("cos", "")
    expr = expr.replace("tan", "")
    expr = expr.replace("log", "")
    expr = expr.replace("exp", "")
    return len(set(x for x in expr if x.isalpha()))


def _should_allow_sympy(expr: str) -> bool:
    """Check if expression is safe to parse with sympy."""
    # Don't parse expressions with too many unknowns
    if _count_unknown_letters(expr) > 2:
        return False

    for bad in _BAD_SUBSTRINGS:
        if bad in expr:
            return False

    for bad_regex in _BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_sympy(gold: str, pred: str) -> bool:
    """Check if two expressions are equal using sympy simplification.

    Returns True if (gold - pred) simplifies to 0.

    Args:
        gold: Ground truth expression
        pred: Predicted expression

    Returns:
        True if expressions are symbolically equivalent
    """
    try:
        import sympy
    except ImportError:
        return False

    try:
        expr = f"({gold})-({pred})"
        if _should_allow_sympy(expr):
            sympy_diff = _sympy_parse(expr)
            if sympy_diff is not None:
                simplified = sympy.simplify(sympy_diff)
                return simplified == 0
    except Exception:
        pass

    return False


def _is_frac_string(expr: str) -> bool:
    """Check if string is a simple fraction like '3/4'."""
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _split_tuple(expr: str) -> list[str]:
    """Split tuple/interval elements, handling commas in numbers."""
    expr = _strip_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in _TUPLE_CHARS
        and expr[-1] in _TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in _TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


def verify_sympy(gold: str | None, pred: str | None) -> bool:
    """Full sympy-based verification with tuple and fraction handling.

    Combines:
    1. Direct string comparison after normalization
    2. Sympy symbolic equivalence checking
    3. Tuple element-wise comparison

    Args:
        gold: Ground truth answer
        pred: Predicted answer

    Returns:
        True if answers are equivalent
    """
    if gold is None or pred is None:
        return False

    gold_norm = _normalize_for_sympy(gold)
    pred_norm = _normalize_for_sympy(pred)

    if gold_norm is None or pred_norm is None:
        return False

    # Direct string match
    if gold_norm == pred_norm:
        return True

    if len(pred_norm) == 0:
        return False

    # Handle tuples/intervals
    gold_elems = _split_tuple(gold_norm)
    pred_elems = _split_tuple(pred_norm)

    # Check tuple brackets match
    if len(gold_elems) > 1 and (
        gold_norm[0] != pred_norm[0] or gold_norm[-1] != pred_norm[-1]
    ):
        return False

    # Check element count
    if len(gold_elems) != len(pred_elems):
        return False

    # Compare element-wise
    for gold_elem, pred_elem in zip(gold_elems, pred_elems):
        # Fractions must match exactly (no simplification for unreduced fracs)
        if _is_frac_string(gold_elem) and _is_frac_string(pred_elem):
            if gold_elem != pred_elem:
                return False
        # Integer ground truth requires exact match
        elif _is_int_string(gold_elem) != _is_int_string(pred_elem):
            return False
        # Otherwise use sympy
        elif not are_equal_sympy(gold_elem, pred_elem):
            return False

    return True


def _is_int_string(s: str) -> bool:
    """Check if string represents an integer."""
    try:
        s = _strip_commas(s)
        val = float(s)
        return abs(val - int(round(val))) <= 1e-7
    except (ValueError, TypeError):
        return False


def normalize_number(s: str | None) -> str | None:
    """Normalize a numeric answer for comparison.

    Handles:
    - Whitespace
    - Commas in numbers
    - Leading/trailing zeros
    - Percentages
    - Fractions (simple cases)

    Args:
        s: Input string

    Returns:
        Normalized string, or None if not a valid number
    """
    if s is None:
        return None

    s = s.strip().replace(",", "").replace(" ", "")

    # Handle percentages
    if s.endswith("%"):
        s = s[:-1]

    # Handle fractions like "1/2"
    if "/" in s and s.count("/") == 1:
        try:
            num, denom = s.split("/")
            return str(float(num) / float(denom))
        except (ValueError, ZeroDivisionError):
            pass

    # Try to parse as float and normalize
    try:
        val = float(s)
        # Remove trailing zeros after decimal
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s.lower()


def extract_boxed(text: str) -> str | None:
    """Extract answer from \\boxed{...} format.

    Args:
        text: Text containing boxed answer

    Returns:
        Content inside boxed, or None if not found
    """
    # Match \boxed{...} with nested braces
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return last match
    return None


def extract_last_number(text: str) -> str | None:
    """Extract the last number from text as fallback.

    Args:
        text: Text to search

    Returns:
        Last number found, or None
    """
    # Match integers and decimals, with optional negative sign
    pattern = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].replace(",", "")
    return None


@dataclass
class GSM8KTask(Task):
    """GSM8K math word problem task."""

    question: str
    gold_answer: str  # The correct numeric answer
    full_solution: str = ""  # Optional: full solution text

    task_type: str = field(default="gsm8k", init=False)

    def to_prompt(self) -> str:
        user = (
            f"{self.question}\n\n"
            "Return ONLY the final numeric answer in the final channel.\n"
            "Use the analysis channel for any reasoning.\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        final = parsed.final_content
        if not final.strip():
            return None
        return extract_last_number(final)

    def verify(self, answer: str | None) -> bool:
        """Verify if answer matches gold."""
        if answer is None:
            return False
        return normalize_number(answer) == normalize_number(self.gold_answer)


@dataclass
class MATHTask(Task):
    """MATH competition problem task.

    Similar to GSM8K but may have more complex answers
    (fractions, expressions, etc.)
    """

    problem: str
    gold_answer: str
    level: int = 1  # Difficulty level 1-5
    subject: str = ""  # e.g., "algebra", "geometry"

    task_type: str = field(default="math", init=False)

    def to_prompt(self) -> str:
        user = (
            f"{self.problem}\n\n"
            "Return ONLY the final answer in the final channel.\n"
            "Use the analysis channel for any reasoning.\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        final = parsed.final_content.strip()
        return final if final else None

    def verify(self, answer: str | None, use_sympy: bool = True) -> bool:
        """Verify answer, handling symbolic equivalence where possible.

        Args:
            answer: Extracted answer to verify
            use_sympy: Use sympy for symbolic equivalence (default True)

        Returns:
            True if answer is correct
        """
        if answer is None:
            return False

        # Try sympy verification first (most robust)
        if use_sympy:
            if verify_sympy(self.gold_answer, answer):
                return True

        # Fall back to numeric comparison
        norm_ans = normalize_number(answer)
        norm_gold = normalize_number(self.gold_answer)

        if norm_ans is not None and norm_gold is not None:
            if norm_ans == norm_gold:
                return True
            # Try float comparison for close values
            try:
                return abs(float(norm_ans) - float(norm_gold)) < 1e-6
            except ValueError:
                pass

        # Final fallback: string comparison
        return answer.strip().lower() == self.gold_answer.strip().lower()

    def get_metadata(self) -> dict:
        return {
            "task_type": self.task_type,
            "level": self.level,
            "subject": self.subject,
        }


# =============================================================================
# Data Loading
# =============================================================================

def _parse_gsm8k_answer(answer_text: str) -> str | None:
    """Parse gold answer from GSM8K answer field.

    GSM8K answers are formatted as:
    "... #### <number>"

    Args:
        answer_text: Full answer text from dataset

    Returns:
        Extracted numeric answer
    """
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) >= 2:
            return parts[-1].strip().replace(",", "")
    return None


def load_gsm8k_tasks(
    max_examples: int = 10000,
    split: str = "train",
    source: str = "openai/gsm8k",
) -> list[GSM8KTask]:
    """Load GSM8K tasks from HuggingFace.

    Args:
        max_examples: Maximum number of examples to load
        split: Dataset split ("train" or "test")
        source: HuggingFace dataset name

    Returns:
        List of GSM8KTask instances
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    ds = load_dataset(source, "main", split=split)

    tasks = []
    for ex in ds:
        if len(tasks) >= max_examples:
            break

        question = ex.get("question", "")
        answer_text = ex.get("answer", "")

        if not question or not answer_text:
            continue

        gold = _parse_gsm8k_answer(answer_text)
        if gold is None:
            continue

        tasks.append(GSM8KTask(
            question=question,
            gold_answer=gold,
            full_solution=answer_text,
        ))

    return tasks


def load_math_tasks(
    max_examples: int = 10000,
    split: str = "train",
    source: str = "hendrycks/competition_math",
    min_level: int = 1,
    max_level: int = 5,
) -> list[MATHTask]:
    """Load MATH competition tasks from HuggingFace.

    Args:
        max_examples: Maximum number of examples
        split: Dataset split
        source: HuggingFace dataset name
        min_level: Minimum difficulty level
        max_level: Maximum difficulty level

    Returns:
        List of MATHTask instances
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    ds = load_dataset(source, split=split)

    tasks = []
    for ex in ds:
        if len(tasks) >= max_examples:
            break

        problem = ex.get("problem", "")
        solution = ex.get("solution", "")
        level_str = ex.get("level", "Level 1")
        subject = ex.get("type", "")

        # Parse level
        level_match = re.search(r"Level (\d+)", level_str)
        level = int(level_match.group(1)) if level_match else 1

        if level < min_level or level > max_level:
            continue

        # Extract gold answer from solution (usually boxed)
        gold = extract_boxed(solution)
        if gold is None:
            continue

        tasks.append(MATHTask(
            problem=problem,
            gold_answer=gold,
            level=level,
            subject=subject,
        ))

    return tasks


def iter_gsm8k(
    max_examples: int = 10000,
    source: str = "hf:openai/gsm8k:main:train",
) -> Iterator[GSM8KTask]:
    """Iterator version for streaming large datasets.

    Args:
        max_examples: Maximum examples to yield
        source: Source spec in format "hf:dataset:subset:split"

    Yields:
        GSM8KTask instances
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    # Parse source spec
    parts = source.split(":")
    if len(parts) >= 4 and parts[0] == "hf":
        dataset_name = parts[1]
        subset = parts[2] if parts[2] else None
        split = parts[3]
    else:
        dataset_name = "openai/gsm8k"
        subset = "main"
        split = "train"

    if subset:
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)

    count = 0
    for ex in ds:
        if count >= max_examples:
            break

        question = ex.get("question", "")
        answer_text = ex.get("answer", "")

        if not question or not answer_text:
            continue

        gold = _parse_gsm8k_answer(answer_text)
        if gold is None:
            continue

        yield GSM8KTask(
            question=question,
            gold_answer=gold,
            full_solution=answer_text,
        )
        count += 1

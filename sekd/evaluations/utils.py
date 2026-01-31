import re
import hashlib
from typing import Any, Dict, Callable

# ==========================================================
#                    Text processing utilities
# ==========================================================

def _strip(s: str) -> str:
    return s.strip().replace(" ", "").replace(",", "")


def _num_from_text(s: str) -> str | None:
    s = str(s).replace("−", "-")
    dollars = re.findall(r"\$([^$]+)\$", s)
    if dollars:
        s = dollars[-1]
    boxed = re.search(r"\\boxed\{([^}]+)\}", s)
    if boxed:
        s = boxed.group(1)
    s = s.replace(",", "")
    match = re.search(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", s)
    return match.group(1) if match else None


def process_results_numeric_em(doc: Dict[str, Any], results: list) -> Dict[str, int]:
    pred_text = results[0] if results else ""
    pred = _num_from_text(pred_text) or _strip(pred_text)
    tgt_raw = str(doc.get("answer", ""))
    tgt = _num_from_text(tgt_raw) or _strip(tgt_raw)
    return {"exact_match": 1 if pred == tgt and pred != "" else 0}


def process_docs_svamp(dataset):
    def _map(row):
        q = row.get("question") or row.get("Body") or ""
        a = str(row.get("answer") or row.get("Answer") or "")
        return {"input": f"Question: {q}\nAnswer:", "answer": a}

    # Handle both DatasetDict and Dataset objects
    if hasattr(dataset, 'keys'):  # DatasetDict
        split = "test" if "test" in dataset else next(iter(dataset.keys()))
        test = dataset[split].map(_map)
        return {"test": test}
    else:  # Single Dataset (when test_split is specified)
        return dataset.map(_map)


def process_docs_aimo(dataset):
    def _map(row):
        prompt = row.get("problem") or row.get("question") or row.get("prompt") or ""
        a = str(row.get("answer") or "")
        body = (
            "Problem:\n"
            f"{prompt}\n\n"
            "Give only the final numeric answer.\n"
            "Answer:"
        )
        return {"input": body, "answer": a}

    # Handle both DatasetDict and Dataset objects
    if hasattr(dataset, 'keys'):  # DatasetDict
        split = "test" if "test" in dataset else next(iter(dataset.keys()))
        test = dataset[split].map(_map)
        return {"test": test}
    else:  # Single Dataset (when test_split is specified)
        return dataset.map(_map)

# utils.py
from typing import Any

def process_docs_ifeval(dataset: Any):
    def _map(row):
        # IFEval rows have 'prompt' (the instruction to follow)
        prompt = row.get("prompt") or ""
        # No gold 'answer' in IFEval; leave empty string
        return {"input": prompt, "answer": ""}

    # Support both DatasetDict and single Dataset (when test_split specified)
    if hasattr(dataset, "keys"):  # DatasetDict
        split = "test" if "test" in dataset else next(iter(dataset.keys()))
        mapped = dataset[split].map(_map)
        return {"test": mapped}
    else:
        return dataset.map(_map)

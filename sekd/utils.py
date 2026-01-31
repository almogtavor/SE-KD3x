# Helper: check if BitsAndBytes + Triton backend are usable in this env
def _bnb_triton_available() -> bool:
    try:
        # transformers integration class
        from transformers import BitsAndBytesConfig  # noqa: F401
    except Exception:
        return False
    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception:
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    # Some bnb builds require triton.ops; probe a representative module
    try:
        import importlib
        importlib.import_module("bitsandbytes.triton.int8_matmul_mixed_dequantize")
    except Exception:
        return False
    return True


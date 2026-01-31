import torch
from typing import Union, Optional, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def load_model(
    model_name: str,
    device_map: Union[str, int, Dict[str, int]] = "auto",
    quant_bits: int = None,
    max_memory: Optional[Dict[Union[int, str], str]] = None,
    force_single_gpu: bool = False,
):
    """Load model with optional quantization to save GPU memory.
    
    Args:
        force_single_gpu: If True, avoid device_map="auto" and load on single GPU with quantization
    """

    # Prefer BF16 on supported GPUs, otherwise fall back to safe defaults.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if force_single_gpu:
        # Force everything to single GPU (GPU 0 for teacher, or specified GPU)
        if isinstance(device_map, int):
            device_map_arg = {"": device_map}
        else:
            device_map_arg = {"": 0}  # Default to GPU 0
        max_memory_arg = None  # Don't use max_memory with single GPU mapping
        print(f"Loading model on single GPU: {device_map_arg} with quantization={quant_bits}-bit")
    else:
        # Legacy multi-GPU mode (only for student model without quantization)
        device_map_arg = device_map if not isinstance(device_map, int) else {"": device_map}
        max_memory_arg = max_memory
        print(f"Loading model with device_map={device_map_arg}, max_memory={max_memory_arg}")
    
    quantization_config = None
    if quant_bits is not None:
        if quant_bits == 4:
            compute_dtype = torch.float16 if torch_dtype == torch.bfloat16 else torch_dtype
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif quant_bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization bit-width: {quant_bits}")
        print(f"Enabling bitsandbytes quantization: {quant_bits}-bit", flush=True)

    print("This may take several minutes for model loading...")

    model_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        device_map=device_map_arg,
        max_memory=max_memory_arg,
        low_cpu_mem_usage=False,  # Disable to avoid hanging issues
        trust_remote_code=True,  # For some models like Qwen
        torch_dtype=torch_dtype,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    print("✅ Model loaded successfully, now loading tokenizer...")

    return model

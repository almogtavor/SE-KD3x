import logging
from typing import Optional

import torch
from ollama import Client
from transformers import AutoTokenizer


class OllamaModel:
    """Wrapper for Ollama models to match HuggingFace interface."""

    def __init__(self, model_name: str = "qwen3:8b"):
        self.client = Client()
        self.model_name = model_name
        # Try to load a compatible tokenizer for tokenization
        try:
            # First try the direct model name
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
            logging.info("Using Qwen2-7B-Instruct tokenizer")
        except Exception as e:
            logging.warning(f"Failed to load Qwen tokenizer: {e}")
            try:
                # Fallback to LLaMA tokenizer which is widely compatible
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                logging.info("Using Llama-2 tokenizer as fallback")
            except Exception as e:
                logging.warning(f"Failed to load Llama tokenizer: {e}")
                # Last resort, use GPT2 tokenizer which is widely available
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                logging.info("Using GPT2 tokenizer as last resort")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text using Ollama model."""
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": 0.0,  # Deterministic for distillation
            }
        )
        return response['response']

    def __call__(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Mimic HuggingFace model interface for distillation."""
        # Convert input_ids to text
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Generate responses
        responses = [self.generate(prompt) for prompt in prompts]

        # Tokenize responses
        response_tokens = self.tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=input_ids.size(1)
        )

        # Create a dummy output object that matches HuggingFace interface
        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits

        # Convert to logits (one-hot for the actual tokens)
        vocab_size = self.tokenizer.vocab_size
        batch_size, seq_len = response_tokens.input_ids.size()
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Set logits to high value for actual tokens
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = response_tokens.input_ids[b, s]
                if token_id != self.tokenizer.pad_token_id:
                    logits[b, s, token_id] = 100.0  # High logit for actual token

        return DummyOutput(logits)

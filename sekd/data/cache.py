"""
Smart caching for FineWeb-Edu and evaluation datasets.
Automatically caches preprocessed data to disk on first use,
then loads from cache on subsequent runs with matching parameters.
"""
import errno
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
from datasets import load_dataset


def _compute_cache_key(
    dataset_name: str,
    tokenizer_name: str,
    max_tokens: Optional[int],
    max_seq_len: int,
    seed: int,
    split: str = "train",
    **extra_params
) -> str:
    """
    Compute a deterministic hash for cache identification.
    Two configs with identical params → same cache key.
    """
    params = {
        "dataset": dataset_name,
        "tokenizer": tokenizer_name,
        "max_tokens": max_tokens,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "split": split,
        **extra_params
    }
    # Sort for determinism
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _get_cache_dir() -> Path:
    """Get the cache directory, preferring fast node-local storage."""
    # Try node-local TMPDIR first (fast on HPC nodes)
    tmpdir = os.environ.get("TMPDIR", None)
    if tmpdir and os.path.exists(tmpdir):
        cache_root = Path(tmpdir) / "ekd_data_cache"
    else:
        # Fallback to workspace
        cache_root = Path("data_cache")
    
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _batched(iterator, size: int):
    """Yield successive size-sized batches from iterator."""
    buf = []
    for ex in iterator:
        buf.append(ex)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


class _ResilientJSONLWriter:
    """Writes JSONL files while recovering from stale file handle errors on NFS."""

    def __init__(
        self,
        path: Path,
        *,
        buffering: int = 1024 * 1024,
        max_retries: int = 5,
        retry_delay: float = 0.2,
    ) -> None:
        self.path = Path(path)
        self.buffering = buffering
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._fh = None
        self._open("w")

    def _open(self, mode: str) -> None:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._fh = open(
                    self.path,
                    mode,
                    encoding="utf-8",
                    buffering=self.buffering,
                )
                return
            except OSError as exc:  # pragma: no cover - re-raised below
                if exc.errno != errno.ESTALE:
                    raise
                last_exc = exc
                time.sleep(self.retry_delay * attempt)
        raise last_exc  # type: ignore[misc]

    def _recover(self) -> None:
        if self._fh:
            try:
                self._fh.close()
            except OSError:
                pass
        print("[data-cache] Detected stale file handle; reopening cache file...")
        self._open("a")

    def _write_line(self, line: str) -> None:
        retries = 0
        while True:
            try:
                self._fh.write(line)
                break
            except OSError as exc:
                retries += 1
                if exc.errno != errno.ESTALE or retries > self.max_retries:
                    raise
                time.sleep(self.retry_delay * retries)
                self._recover()

    def write_examples(self, examples: Iterable[Dict[str, Any]]) -> None:
        for example in examples:
            self._write_line(json.dumps(example, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def load_or_create_fineweb_cache(
    tokenizer,
    max_tokens: int,
    max_seq_len: int,
    seed: int = 1337,
    batch_size: int = 512,
    packing_enabled: bool = True,
) -> List[Dict[str, str]]:
    """
    Load FineWeb-Edu subset from cache if available, otherwise create it.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_tokens: Token budget
        max_seq_len: Reference length for reporting corpus statistics (no filtering is applied)
        seed: Random seed for shuffling
        batch_size: Batch size for tokenization
        packing_enabled: Whether downstream training packs sequences; used in cache key only
    
    Returns:
        List of {"prompt": str, "answer": str} dicts
    """
    # Get tokenizer name for cache key
    tokenizer_name = getattr(tokenizer, "name_or_path", "unknown")
    
    # Compute cache key
    cache_key = _compute_cache_key(
        dataset_name="fineweb-edu",
        tokenizer_name=tokenizer_name,
        max_tokens=max_tokens,
        max_seq_len=max_seq_len,
        seed=seed,
        split="train",
        packing_enabled=packing_enabled,
    )
    
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"fineweb_{cache_key}.jsonl"
    metadata_file = cache_dir / f"fineweb_{cache_key}.meta.json"
    
    # Try to load from cache
    if cache_file.exists() and metadata_file.exists():
        print(f"[data-cache] Loading FineWeb-Edu from cache: {cache_file.name}")
        try:
            examples = []
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    examples.append({"prompt": ex["prompt"], "answer": ex.get("answer", "")})
            
            # Verify metadata
            with open(metadata_file, "r") as f:
                meta = json.load(f)
            
            print(f"[data-cache] Loaded {len(examples):,} docs, {meta['total_tokens']:,} tokens from cache")
            print(
                "[data-cache] Cache params: max_seq_len={}, seed={}, packing_enabled={}".format(
                    meta.get("max_seq_len"),
                    meta.get("seed"),
                    meta.get("packing_enabled", True),
                )
            )
            return examples
        except Exception as e:
            print(f"[data-cache] Cache load failed ({e}), regenerating...")
    
    # Check for incomplete cache (exists but no metadata - from interrupted run)
    if cache_file.exists() and not metadata_file.exists():
        print(f"[data-cache] Found incomplete cache from interrupted run: {cache_file.name}")
        try:
            # Try to load and validate
            examples = []
            total_tokens = 0
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    examples.append({"prompt": ex["prompt"], "answer": ex.get("answer", "")})
                    total_tokens += ex.get("tokens", 0)
            
            # If we have enough data (>= 90% of target), use it and create metadata
            if total_tokens >= max_tokens * 0.9:
                print(f"[data-cache] Incomplete cache has {total_tokens:,} tokens ({total_tokens/max_tokens*100:.1f}% of target)")
                print(f"[data-cache] Salvaging {len(examples):,} docs from interrupted run...")
                
                # Create metadata for salvaged cache
                metadata = {
                    "dataset": "fineweb-edu",
                    "tokenizer": tokenizer_name,
                    "max_tokens": max_tokens,
                    "max_seq_len": max_seq_len,
                    "seed": seed,
                    "total_tokens": total_tokens,
                    "num_docs": len(examples),
                    "salvaged": True,
                    "cache_file": str(cache_file),
                    "packing_enabled": packing_enabled,
                }
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                print("[data-cache] ✓ Salvaged cache validated and ready to use")
                return [{"prompt": ex["prompt"], "answer": ex["answer"]} for ex in examples]
            else:
                print(f"[data-cache] Incomplete cache only has {total_tokens:,} tokens ({total_tokens/max_tokens*100:.1f}% of target)")
                print("[data-cache] Discarding and regenerating...")
                cache_file.unlink()
        except Exception as e:
            print(f"[data-cache] Failed to salvage incomplete cache ({e}), regenerating...")
            if cache_file.exists():
                cache_file.unlink()
    
    # Cache miss - create from scratch with batched tokenization
    print(f"[data-cache] Creating FineWeb-Edu cache: {cache_file.name}")
    print(
        "[data-cache] Params: max_tokens={:,}, max_seq_len={}, seed={}, packing_enabled={}".format(
            max_tokens,
            max_seq_len,
            seed,
            packing_enabled,
        )
    )
    
    # Load streaming dataset (num_proc not supported for streaming)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    
    total_tokens = 0
    examples = []
    docs_seen = 0
    docs_long_total = 0
    docs_short_total = 0
    batches_processed = 0
    budget_reached = False
    write_buffer = []  # Buffer for batched writes (faster disk I/O)
    write_buffer_size = 1000
    
    writer = _ResilientJSONLWriter(cache_file)
    try:
        for batch in _batched(ds, batch_size):
            if budget_reached:
                break

            batches_processed += 1
            texts = [ex.get("text", "") for ex in batch if ex.get("text")]
            docs_seen += len(texts)

            if batches_processed % 50 == 0:  # Reduced logging frequency (20->50) for speed
                print(
                    f"[data-cache] Batch {batches_processed}: processed {docs_seen:,} docs, "
                    f"kept {len(examples):,}, tokens: {total_tokens:,}"
                )

            if not texts:
                continue

            # Batch tokenize (much faster!) - with padding for consistent tensor shapes (faster on GPU tokenizers)
            enc = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)

            for txt, ids in zip(texts, enc["input_ids"]):
                n_tokens = len(ids)

                # Filter by length
                if n_tokens > max_seq_len:
                    docs_long_total += 1
                elif n_tokens < max_seq_len:
                    docs_short_total += 1

                # Check token budget - stop AFTER we've reached the budget
                if total_tokens >= max_tokens:
                    budget_reached = True
                    break

                ex = {
                    "prompt": txt,
                    "answer": "",
                    "tokens": n_tokens,
                }
                examples.append(ex)
                write_buffer.append(ex)
                total_tokens += n_tokens

                # Flush write buffer periodically (batched disk writes are much faster)
                if len(write_buffer) >= write_buffer_size:
                    writer.write_examples(write_buffer)
                    write_buffer.clear()

        # Final flush
        if write_buffer:
            writer.write_examples(write_buffer)
    finally:
        writer.close()
    
    print(f"[data-cache] Cache file written ({len(examples):,} docs)")
    
    # Save metadata
    metadata = {
        "dataset": "fineweb-edu",
        "tokenizer": tokenizer_name,
        "max_tokens": max_tokens,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "total_tokens": total_tokens,
        "num_docs": len(examples),
        "docs_seen": docs_seen,
        "docs_long_total": docs_long_total,
        "docs_short_total": docs_short_total,
        "packing_enabled": packing_enabled,
        "long_doc_rate": docs_long_total / docs_seen if docs_seen > 0 else 0.0,
        "short_doc_rate": docs_short_total / docs_seen if docs_seen > 0 else 0.0,
        "cache_file": str(cache_file),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    file_size_mb = cache_file.stat().st_size / 1024**2
    print(f"[data-cache] ✓ Cache created: {file_size_mb:.1f} MB")
    if docs_seen > 0:
        if docs_seen > 0:
            long_rate_pct = docs_long_total / docs_seen * 100
            short_rate_pct = docs_short_total / docs_seen * 100
            print(
                "[data-cache] Docs > max_seq_len: {}/{} ({:.1f}%), docs < max_seq_len: {}/{} ({:.1f}%)".format(
                    docs_long_total,
                    docs_seen,
                    long_rate_pct,
                    docs_short_total,
                    docs_seen,
                    short_rate_pct,
                )
            )
    else:
        print("[data-cache] Dataset stream was empty; no filtering stats available")
    
    # Return without 'tokens' field (just prompt/answer)
    return [{"prompt": ex["prompt"], "answer": ex["answer"]} for ex in examples]


def load_or_create_eval_cache(
    dataset_name: str,
    split: str,
    tokenizer,
    max_seq_len: int,
    dataset_config: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 256,
) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from cache if available, otherwise create it.
    No length-based filtering is applied; statistics about over-length examples are recorded instead.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "gsm8k")
        split: Dataset split (e.g., "test")
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
        dataset_config: Optional dataset config (e.g., "main" for gsm8k)
        max_samples: Optional limit on number of samples
        batch_size: Batch size for tokenization
    
    Returns:
        List of dataset examples (format depends on dataset)
    """
    # Get tokenizer name for cache key
    tokenizer_name = getattr(tokenizer, "name_or_path", "unknown")
    
    # Compute cache key
    cache_key = _compute_cache_key(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_tokens=None,  # eval doesn't use token budget
        max_seq_len=max_seq_len,
        seed=0,  # eval is deterministic
        split=split,
        config=dataset_config,
        max_samples=max_samples,
    )
    
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"eval_{dataset_name.replace('/', '_')}_{cache_key}.jsonl"
    metadata_file = cache_dir / f"eval_{dataset_name.replace('/', '_')}_{cache_key}.meta.json"
    
    # Try to load from cache
    if cache_file.exists() and metadata_file.exists():
        print(f"[eval-cache] Loading {dataset_name} from cache: {cache_file.name}")
        try:
            examples = []
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    examples.append(json.loads(line))
            
            print(f"[eval-cache] Loaded {len(examples):,} examples from cache")
            return examples
        except Exception as e:
            print(f"[eval-cache] Cache load failed ({e}), regenerating...")
    
    # Cache miss - create from scratch
    print(f"[eval-cache] Creating cache for {dataset_name} ({split}): {cache_file.name}")
    print(f"[eval-cache] Params: max_seq_len={max_seq_len}, config={dataset_config}")
    
    # Load dataset
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    
    # Apply max_samples if specified
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    
    # Filter by length (batch processing)
    examples = []
    docs_long = 0
    
    for batch_start in range(0, len(ds), batch_size):
        batch = ds[batch_start:min(batch_start + batch_size, len(ds))]
        
        # Get text representation (try common fields)
        texts = []
        for i in range(len(batch[list(batch.keys())[0]])):
            ex = {k: v[i] for k, v in batch.items()}
            # Try to construct text from common fields
            text_parts = []
            for field in ["question", "prompt", "text", "input"]:
                if field in ex and ex[field]:
                    text_parts.append(str(ex[field]))
            for field in ["answer", "completion", "output"]:
                if field in ex and ex[field]:
                    text_parts.append(str(ex[field]))
            
            text = " ".join(text_parts) if text_parts else str(ex)
            texts.append((text, ex))
        
        # Batch tokenize
        enc = tokenizer([t[0] for t in texts], add_special_tokens=False)
        
        for (text, ex), ids in zip(texts, enc["input_ids"]):
            n_tokens = len(ids)
            
            # Store original example with token count
            ex["_tokens"] = n_tokens
            examples.append(ex)
            if n_tokens > max_seq_len:
                docs_long += 1
    
    # Save to cache
    print(f"[eval-cache] Saving {len(examples):,} examples to cache...")
    with open(cache_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "config": dataset_config,
        "split": split,
        "tokenizer": tokenizer_name,
        "max_seq_len": max_seq_len,
        "max_samples": max_samples,
        "num_examples": len(examples),
        "total_examples": len(ds),
        "docs_over_max_seq_len": docs_long,
        "over_length_rate": docs_long / len(ds) if len(ds) > 0 else 0.0,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    over_pct = (docs_long / len(ds) * 100) if len(ds) > 0 else 0.0
    print(f"[eval-cache] ✓ Cache created: {len(examples):,}/{len(ds):,} examples")
    print(f"[eval-cache] Docs exceeding {max_seq_len} tokens: {docs_long} ({over_pct:.1f}%)")
    
    return examples

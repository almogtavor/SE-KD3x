import errno
import json

from sekd.data.cache import _ResilientJSONLWriter


class _FlakyHandle:
    """Proxy that simulates a transient ESTALE error on write."""

    def __init__(self, wrapped, failures: int = 1):
        self._wrapped = wrapped
        self._failures = failures

    def write(self, data):
        if self._failures > 0:
            self._failures -= 1
            raise OSError(errno.ESTALE, "Stale file handle")
        return self._wrapped.write(data)

    def close(self):
        return self._wrapped.close()


def test_resilient_writer_recovers_from_stale_handle(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    records = [
        {"prompt": "a", "answer": "", "tokens": 1},
        {"prompt": "b", "answer": "", "tokens": 2},
    ]
    writer = _ResilientJSONLWriter(cache_file, max_retries=3, retry_delay=0.0)
    writer._fh = _FlakyHandle(writer._fh, failures=1)

    try:
        writer.write_examples(records)
    finally:
        writer.close()

    stored = [json.loads(line) for line in cache_file.read_text(encoding="utf-8").splitlines()]
    assert stored == records

"""Entry point that optionally enables our FSDP patch before delegating to lm-eval."""

from __future__ import annotations

import sys

from sekd.evaluations.fsdp_patch import maybe_enable_fsdp


def main() -> int:
    """Apply FSDP instrumentation (if requested) then invoke lm-eval's CLI."""
    maybe_enable_fsdp()
    from lm_eval.__main__ import cli_evaluate  # imported lazily after patch

    return cli_evaluate()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

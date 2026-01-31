"""Command-line interface for SE-KD3X distillation."""

import sys


def main():
    """Main entry point - delegates to run_distillation.py."""
    # Import here to avoid circular imports
    import runpy
    from pathlib import Path

    # Find the run_distillation.py script
    script_path = Path(__file__).parent.parent / "run_distillation.py"

    if script_path.exists():
        sys.argv[0] = str(script_path)
        runpy.run_path(str(script_path), run_name="__main__")
    else:
        print("Error: run_distillation.py not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

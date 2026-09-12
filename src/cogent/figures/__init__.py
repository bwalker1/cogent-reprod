"""Spatial gene embedding figure analyses."""


def arguments():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run gene embedding figure analyses")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()

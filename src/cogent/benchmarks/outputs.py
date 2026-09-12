"""Keep comparator seeds and training modes in separate output stores."""

from pathlib import Path


def output_paths(output_dir, method, seeds, mode=None):
    """Retain Figure 3's filenames for seed 0 and scETM's per-stage mode."""
    if len(set(seeds)) != len(seeds):
        raise ValueError("Provide each seed only once")
    paths = []
    for seed in seeds:
        canonical = seed == 0 and mode in (None, "per-stage")
        suffix = "" if canonical else f"-{mode + '-' if mode else ''}seed{seed}"
        path = Path(output_dir) / f"{method}{suffix}.zarr"
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}. Choose a new --output-dir."
            )
        paths.append(path)
    return paths

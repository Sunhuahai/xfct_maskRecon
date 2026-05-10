from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "projections"
    / "mask"
    / "geometry_45_proj_cmask9_grid_p6_d1d25.npy"
)


def angle_indices(full_angle_count: int, target_angle_count: int) -> np.ndarray:
    if target_angle_count <= 0:
        raise ValueError("target_angle_count must be positive.")
    if full_angle_count % target_angle_count != 0:
        raise ValueError(
            f"{full_angle_count} angles cannot be evenly subsampled to {target_angle_count}."
        )
    step = full_angle_count // target_angle_count
    return np.arange(0, full_angle_count, step, dtype=int)


def output_path_for(input_path: Path, target_angle_count: int) -> Path:
    name = input_path.name
    if "_45_" in name:
        return input_path.with_name(name.replace("_45_", f"_{target_angle_count}_", 1))
    return input_path.with_name(f"{input_path.stem}_{target_angle_count}angles{input_path.suffix}")


def subsample(input_path: Path, target_counts: list[int]) -> list[Path]:
    projection = np.load(input_path)
    if projection.ndim != 3:
        raise ValueError(f"Expected [angle, z, x] projection, got {projection.shape}.")

    outputs: list[Path] = []
    full_angle_count = int(projection.shape[0])
    for target in target_counts:
        indices = angle_indices(full_angle_count, int(target))
        out = output_path_for(input_path, int(target))
        np.save(out, projection[indices])
        outputs.append(out)
        print(f"{target} angles: indices={indices.tolist()} -> {out}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniformly subsample a 45-angle coded-mask projection to 5/15 angles."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input 45-angle .npy file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--targets",
        default="5,15",
        help="Comma-separated target angle counts. Default: 5,15.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = [int(item.strip()) for item in args.targets.split(",") if item.strip()]
    subsample(Path(args.input), targets)


if __name__ == "__main__":
    main()

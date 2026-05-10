from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mask_xfct_model import format_float_tag


def _sanitize(text: str) -> str:
    return (
        str(text)
        .replace(".", "d")
        .replace("-", "m")
        .replace("+", "p")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _pairwise_min_distance(centers: np.ndarray) -> float:
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    if centers.shape[0] <= 1:
        return float("inf")
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    dist[dist == 0.0] = np.inf
    return float(np.min(dist))


def _inside_radius(centers: np.ndarray, mask_radius_mm: float) -> bool:
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    return bool(np.all(np.sqrt(np.sum(centers**2, axis=1)) <= float(mask_radius_mm) + 1.0e-9))


def _current_grid3x3(pitch_mm: float) -> np.ndarray:
    coords = (np.arange(3, dtype=float) - 1.0) * float(pitch_mm)
    xx, zz = np.meshgrid(coords, coords, indexing="xy")
    return np.column_stack([xx.ravel(), zz.ravel()])


def _cross_plus_center(pitch_mm: float) -> np.ndarray:
    p = float(pitch_mm)
    return np.array([[0.0, 0.0], [-p, 0.0], [p, 0.0], [0.0, -p], [0.0, p]], dtype=float)


def _ring(n_holes: int, min_distance_mm: float, mask_radius_mm: float, two_ring: bool = False) -> np.ndarray:
    n = int(n_holes)
    if n <= 1:
        return np.array([[0.0, 0.0]], dtype=float)
    if two_ring and n >= 7:
        inner_n = max(3, n // 3)
        outer_n = n - inner_n - 1
        inner_radius = max(float(min_distance_mm), 0.33 * float(mask_radius_mm))
        outer_radius = min(float(mask_radius_mm), max(inner_radius + float(min_distance_mm), 0.75 * float(mask_radius_mm)))
        centers = [[0.0, 0.0]]
        for k in range(inner_n):
            theta = 2.0 * math.pi * k / inner_n + math.pi / inner_n
            centers.append([inner_radius * math.cos(theta), inner_radius * math.sin(theta)])
        for k in range(outer_n):
            theta = 2.0 * math.pi * k / max(outer_n, 1)
            centers.append([outer_radius * math.cos(theta), outer_radius * math.sin(theta)])
        return np.asarray(centers[:n], dtype=float)
    ring_n = n - 1
    radius_min = float(min_distance_mm) / max(2.0 * math.sin(math.pi / max(ring_n, 2)), 1.0e-9)
    radius = min(float(mask_radius_mm), max(radius_min, 0.55 * float(mask_radius_mm)))
    centers = [[0.0, 0.0]]
    for k in range(ring_n):
        theta = 2.0 * math.pi * k / ring_n
        centers.append([radius * math.cos(theta), radius * math.sin(theta)])
    return np.asarray(centers, dtype=float)


def _random_sparse(n_holes: int, min_distance_mm: float, mask_radius_mm: float, rng: np.random.Generator) -> np.ndarray | None:
    centers: list[np.ndarray] = [np.array([0.0, 0.0], dtype=float)]
    attempts = 0
    while len(centers) < int(n_holes) and attempts < 5000:
        attempts += 1
        radius = float(mask_radius_mm) * math.sqrt(float(rng.random()))
        theta = 2.0 * math.pi * float(rng.random())
        candidate = np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=float)
        if all(float(np.linalg.norm(candidate - c)) >= float(min_distance_mm) for c in centers):
            centers.append(candidate)
    if len(centers) != int(n_holes):
        return None
    return np.asarray(centers, dtype=float)


def _blue_noise(n_holes: int, min_distance_mm: float, mask_radius_mm: float, seed: int) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    # For this small mask, repeated dart throwing gives a stable Poisson-disk-like set.
    for _ in range(80):
        candidate = _random_sparse(n_holes, min_distance_mm, mask_radius_mm, rng)
        if candidate is None:
            continue
        radii = np.sqrt(np.sum(candidate**2, axis=1))
        spread = float(np.std(radii) + 0.15 * np.mean(radii) + _pairwise_min_distance(candidate))
        if spread > best_score:
            best = candidate
            best_score = spread
    return best


def _ura_sparse(n_holes: int, pitch_mm: float, mask_radius_mm: float) -> np.ndarray | None:
    # A sparse cyclic-difference-inspired subset on a 5x5 lattice. This is kept
    # as a diagnostic baseline; no fixed-shift URA decoding is implied.
    grid = []
    p = float(pitch_mm)
    for iz in range(-2, 3):
        for ix in range(-2, 3):
            if (ix * ix + iz * iz) == 0:
                grid.append((0.0, 0.0, 0))
                continue
            if (ix * ix + 2 * iz * iz + ix * iz) % 5 in {0, 1}:
                grid.append((ix * p, iz * p, ix * ix + iz * iz))
    grid = sorted(grid, key=lambda item: (item[2], abs(item[0]) + abs(item[1])))
    centers = []
    seen = set()
    for x, z, _ in grid:
        key = (round(x, 8), round(z, 8))
        if key in seen:
            continue
        if math.hypot(x, z) <= float(mask_radius_mm) + 1.0e-9:
            centers.append([x, z])
            seen.add(key)
        if len(centers) >= int(n_holes):
            break
    if len(centers) != int(n_holes):
        return None
    if [0.0, 0.0] not in centers:
        centers[0] = [0.0, 0.0]
    return np.asarray(centers, dtype=float)


def _candidate_id(family: str, centers: np.ndarray, diameter: float, min_distance: float, seed: int | None = None) -> str:
    seed_tag = "" if seed is None else f"_s{int(seed)}"
    n = int(np.asarray(centers).shape[0])
    return (
        f"{_sanitize(family)}_n{n}_d{format_float_tag(diameter)}_"
        f"mind{format_float_tag(min_distance)}{seed_tag}"
    )


def _make_candidate(
    *,
    family: str,
    centers: np.ndarray,
    diameter: float,
    min_distance: float,
    mask_radius: float,
    comment: str,
    seed: int | None = None,
) -> dict | None:
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    if not _inside_radius(centers, mask_radius):
        return None
    actual_min = _pairwise_min_distance(centers)
    if centers.shape[0] > 1 and actual_min + 1.0e-9 < float(min_distance):
        return None
    candidate_id = _candidate_id(family, centers, diameter, min_distance, seed=seed)
    return {
        "candidate_id": candidate_id,
        "family": str(family),
        "hole_centers_mm": [[float(x), float(z)] for x, z in centers],
        "hole_diameter_mm": float(diameter),
        "min_distance_mm": float(min_distance if np.isfinite(min_distance) else 0.0),
        "actual_min_distance_mm": float(actual_min if np.isfinite(actual_min) else 0.0),
        "mask_radius_mm": float(mask_radius),
        "total_open_area_mm2": float(centers.shape[0] * math.pi * (float(diameter) / 2.0) ** 2),
        "comments": str(comment),
    }


def generate_candidates(args: argparse.Namespace) -> list[dict]:
    if args.quick:
        hole_counts = [1, 3, 5, 7, 9]
        diameters = [0.75, 1.25]
        min_distances = [3.0, 6.0]
        seeds = [0, 1]
    else:
        hole_counts = [1, 3, 5, 7, 9]
        diameters = [0.5, 0.75, 1.0, 1.25, 1.5]
        min_distances = [2.0, 3.0, 4.0, 5.0, 6.0]
        seeds = [0, 1, 2, 3]
    mask_radius = float(args.mask_radius_mm)
    candidates: list[dict] = []

    for diameter in diameters:
        cand = _make_candidate(
            family="single_center",
            centers=np.array([[0.0, 0.0]], dtype=float),
            diameter=diameter,
            min_distance=0.0,
            mask_radius=mask_radius,
            comment="single center pinhole regression/control",
        )
        if cand:
            candidates.append(cand)

    for diameter in diameters:
        centers = _current_grid3x3(6.0)
        cand = _make_candidate(
            family="grid3x3",
            centers=centers,
            diameter=diameter,
            min_distance=6.0,
            mask_radius=mask_radius,
            comment="current grid baseline" if abs(diameter - 1.25) < 1.0e-9 else "grid3x3 diagnostic baseline",
        )
        if cand:
            candidates.append(cand)

    for diameter in diameters:
        for min_distance in min_distances:
            centers = _cross_plus_center(min_distance)
            cand = _make_candidate(
                family="cross_plus_center",
                centers=centers,
                diameter=diameter,
                min_distance=min_distance,
                mask_radius=mask_radius,
                comment="5-hole controlled-overlap baseline",
            )
            if cand:
                candidates.append(cand)

    for n_holes in hole_counts:
        if n_holes == 1:
            continue
        for diameter in diameters:
            for min_distance in min_distances:
                for two_ring in ([False, True] if n_holes >= 7 else [False]):
                    centers = _ring(n_holes, min_distance, mask_radius, two_ring=two_ring)
                    family = "ring_two" if two_ring else "ring"
                    cand = _make_candidate(
                        family=family,
                        centers=centers,
                        diameter=diameter,
                        min_distance=min_distance,
                        mask_radius=mask_radius,
                        comment=f"{n_holes}-hole ring sparse depth-coded candidate",
                    )
                    if cand:
                        candidates.append(cand)

    for n_holes in hole_counts:
        if n_holes == 1:
            continue
        for diameter in diameters:
            for min_distance in min_distances:
                for seed in seeds:
                    centers = _random_sparse(
                        n_holes=n_holes,
                        min_distance_mm=min_distance,
                        mask_radius_mm=mask_radius,
                        rng=np.random.default_rng(int(args.seed) + 1000 * seed + 17 * n_holes),
                    )
                    if centers is not None:
                        cand = _make_candidate(
                            family="sparse_random",
                            centers=centers,
                            diameter=diameter,
                            min_distance=min_distance,
                            mask_radius=mask_radius,
                            comment=f"random sparse {n_holes}-hole candidate with minimum spacing",
                            seed=seed,
                        )
                        if cand:
                            candidates.append(cand)
                    centers = _blue_noise(
                        n_holes=n_holes,
                        min_distance_mm=min_distance,
                        mask_radius_mm=mask_radius,
                        seed=int(args.seed) + 2000 * seed + 31 * n_holes,
                    )
                    if centers is not None:
                        cand = _make_candidate(
                            family="blue_noise",
                            centers=centers,
                            diameter=diameter,
                            min_distance=min_distance,
                            mask_radius=mask_radius,
                            comment=f"blue-noise sparse {n_holes}-hole candidate",
                            seed=seed,
                        )
                        if cand:
                            candidates.append(cand)
                centers = _ura_sparse(n_holes, min_distance, mask_radius)
                if centers is not None:
                    cand = _make_candidate(
                        family="ura_mura_inspired",
                        centers=centers,
                        diameter=diameter,
                        min_distance=min_distance,
                        mask_radius=mask_radius,
                        comment="URA/MURA-inspired sparse-center diagnostic baseline; no fixed-shift decoding",
                    )
                    if cand:
                        candidates.append(cand)

    dedup: dict[str, dict] = {}
    for cand in candidates:
        dedup.setdefault(cand["candidate_id"], cand)
    ordered = sorted(dedup.values(), key=lambda c: (c["family"], len(c["hole_centers_mm"]), c["hole_diameter_mm"], c["min_distance_mm"], c["candidate_id"]))
    if args.candidate_limit is not None:
        ordered = ordered[: int(args.candidate_limit)]
    return ordered


def write_candidates(candidates: list[dict], args: argparse.Namespace) -> None:
    candidate_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for cand in candidates:
        path = candidate_dir / f"{cand['candidate_id']}.json"
        path.write_text(json.dumps(cand, indent=2), encoding="utf-8")
    fieldnames = [
        "candidate_id",
        "family",
        "hole_count",
        "hole_diameter_mm",
        "min_distance_mm",
        "actual_min_distance_mm",
        "mask_radius_mm",
        "total_open_area_mm2",
        "hole_centers_mm",
        "comments",
        "json_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            row = dict(cand)
            row["hole_count"] = len(cand["hole_centers_mm"])
            row["hole_centers_mm"] = json.dumps(cand["hole_centers_mm"])
            row["json_path"] = str(candidate_dir / f"{cand['candidate_id']}.json")
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sparse multi-pinhole XFCT mask candidate JSON files.")
    parser.add_argument("--quick", action="store_true", help="Generate a reduced deterministic candidate set.")
    parser.add_argument("--final", action="store_true", help="Accepted for workflow compatibility; full mode is default.")
    parser.add_argument("--num-seeds", type=int, default=1, help="Accepted for workflow compatibility.")
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--protocols", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--recon-methods", default="", help="Accepted for workflow compatibility.")
    parser.add_argument("--matrix-mode", choices=["explicit", "matrix_free", "auto"], default="auto")
    parser.add_argument("--mask-radius-mm", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "masks" / "candidates"))
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "results" / "mask_design" / "candidate_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = generate_candidates(args)
    write_candidates(candidates, args)
    print(f"Generated {len(candidates)} mask candidates.")
    print(f"Candidate JSON directory: {args.output_dir}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()

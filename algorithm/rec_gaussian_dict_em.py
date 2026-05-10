from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.positive_gaussian_dictionary import (
    create_gaussian_atoms,
    reconstruct_from_amplitudes,
)

EPS = 1e-10


def _load_projection_dictionary(path: str | Path) -> np.ndarray:
    cache_path = Path(path)
    if cache_path.suffix == ".npz":
        with np.load(cache_path) as data:
            return np.asarray(data["P"], dtype=np.float64)
    if cache_path.suffix == ".npy":
        return np.asarray(np.load(cache_path), dtype=np.float64)
    raise ValueError("projection dictionary cache path must end in .npz or .npy.")


def _save_projection_dictionary(path: str | Path, P: np.ndarray) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix == ".npz":
        np.savez_compressed(cache_path, P=np.asarray(P, dtype=np.float32))
    elif cache_path.suffix == ".npy":
        np.save(cache_path, np.asarray(P, dtype=np.float32))
    else:
        raise ValueError("projection dictionary cache path must end in .npz or .npy.")


def _project_atom_chunk(projector, atom_chunk: np.ndarray) -> np.ndarray:
    flat = atom_chunk.reshape(atom_chunk.shape[0], -1).astype(np.float64, copy=False)
    if hasattr(projector, "shape") and hasattr(projector, "__matmul__"):
        projected = projector @ flat.T
        return np.asarray(projected, dtype=np.float64)
    if callable(projector):
        columns = [np.asarray(projector(atom), dtype=np.float64).ravel() for atom in atom_chunk]
        return np.column_stack(columns)
    raise TypeError("projector must be a system matrix with @ or a callable.")


def build_projection_dictionary(
    atoms: np.ndarray,
    measured_angles=None,
    projector=None,
    chunk_size: int = 32,
    cache_path: str | Path | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """Build P[:, k] = A_meas phi_k with chunked atom projection."""
    if projector is None:
        raise ValueError("projector is required to build the projection dictionary.")
    if cache_path is not None and Path(cache_path).exists():
        return _load_projection_dictionary(cache_path)

    atom_array = np.asarray(atoms)
    if atom_array.ndim != 4:
        raise ValueError(f"atoms must have shape [K, Z, Y, X], got {atom_array.shape}.")
    if measured_angles is not None:
        np.asarray(measured_angles, dtype=np.float64).reshape(-1)

    k_atoms = atom_array.shape[0]
    chunk_size = max(1, int(chunk_size))
    first = _project_atom_chunk(projector, atom_array[:1])
    if first.ndim != 2 or first.shape[1] != 1:
        raise ValueError(f"projected atom chunk must have shape [M, K], got {first.shape}.")

    P = np.empty((first.shape[0], k_atoms), dtype=dtype)
    P[:, :1] = np.maximum(first, 0.0).astype(dtype, copy=False)
    for start in range(1, k_atoms, chunk_size):
        stop = min(start + chunk_size, k_atoms)
        projected = _project_atom_chunk(projector, atom_array[start:stop])
        if projected.shape != (P.shape[0], stop - start):
            raise ValueError(
                "projected chunk shape mismatch: "
                f"expected {(P.shape[0], stop - start)}, got {projected.shape}."
            )
        P[:, start:stop] = np.maximum(projected, 0.0).astype(dtype, copy=False)

    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    if cache_path is not None:
        _save_projection_dictionary(cache_path, P)
    return P


def _background_vector(b: float | np.ndarray | None, size: int) -> np.ndarray:
    if b is None:
        return np.zeros(size, dtype=np.float64)
    if np.isscalar(b):
        return np.full(size, max(float(b), 0.0), dtype=np.float64)
    background = np.asarray(b, dtype=np.float64).reshape(-1)
    if background.size != size:
        raise ValueError(f"background length {background.size} does not match y size {size}.")
    return np.maximum(np.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def poisson_objective(
    P: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    b: float | np.ndarray | None = None,
    lambda_l1: float = 0.0,
) -> float:
    """Return Poisson KL data divergence plus lambda_l1 * sum_k a_k."""
    matrix = np.asarray(P, dtype=np.float64)
    amplitudes = np.maximum(np.asarray(a, dtype=np.float64).reshape(-1), 0.0)
    counts = np.maximum(np.asarray(y, dtype=np.float64).reshape(-1), 0.0)
    background = _background_vector(b, counts.size)
    if matrix.shape != (counts.size, amplitudes.size):
        raise ValueError(
            f"P shape {matrix.shape} is incompatible with y/a sizes "
            f"{counts.size}/{amplitudes.size}."
        )

    mu = np.maximum(matrix @ amplitudes + background, EPS)
    positive = counts > 0.0
    divergence = np.sum(mu - counts)
    divergence += np.sum(counts[positive] * np.log(counts[positive] / mu[positive]))
    penalty = max(float(lambda_l1), 0.0) * float(np.sum(amplitudes))
    return float(divergence + penalty)


def _initial_amplitudes(
    P: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    init: str | np.ndarray,
) -> np.ndarray:
    k_atoms = P.shape[1]
    sensitivity = np.maximum(np.sum(P, axis=0), EPS)
    if isinstance(init, str):
        mode = init.strip().lower()
        corrected_total = max(float(np.sum(y - b)), EPS)
        if mode == "uniform":
            value = corrected_total / (float(np.sum(sensitivity)) + EPS)
            return np.full(k_atoms, max(value, EPS), dtype=np.float64)
        if mode == "matched":
            score = P.T @ np.maximum(y - b, 0.0)
            amplitudes = score / sensitivity
            mean_positive = float(np.mean(amplitudes[amplitudes > 0.0])) if np.any(
                amplitudes > 0.0
            ) else corrected_total / (float(np.sum(sensitivity)) + EPS)
            return np.maximum(amplitudes, max(mean_positive, EPS) * 1.0e-3)
        raise ValueError("init must be 'uniform', 'matched', or an amplitude array.")
    amplitudes = np.asarray(init, dtype=np.float64).reshape(-1)
    if amplitudes.size != k_atoms:
        raise ValueError(f"init amplitude length {amplitudes.size} does not match K={k_atoms}.")
    return np.maximum(amplitudes, EPS)


def poisson_em_amplitude_fit(
    P: np.ndarray,
    y: np.ndarray,
    b: float | np.ndarray | None = None,
    lambda_l1: float = 0.0,
    iterations: int = 100,
    init: str | np.ndarray = "uniform",
    eps: float = EPS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit nonnegative Gaussian amplitudes with penalized Poisson EM updates."""
    matrix = np.maximum(np.asarray(P, dtype=np.float64), 0.0)
    counts = np.maximum(np.asarray(y, dtype=np.float64).reshape(-1), 0.0)
    if matrix.ndim != 2:
        raise ValueError(f"P must be 2D, got shape {matrix.shape}.")
    if matrix.shape[0] != counts.size:
        raise ValueError(f"P row count {matrix.shape[0]} does not match y size {counts.size}.")
    background = _background_vector(b, counts.size)
    lam = max(float(lambda_l1), 0.0)
    n_iter = max(0, int(iterations))

    amplitudes = _initial_amplitudes(matrix, counts, background, init)
    denominator = np.sum(matrix, axis=0) + lam + float(eps)
    objective = [poisson_objective(matrix, amplitudes, counts, background, lam)]
    rel_change = [0.0]
    projection_nrmse = [
        float(
            np.linalg.norm((matrix @ amplitudes + background) - counts)
            / (np.linalg.norm(counts) + float(eps))
        )
    ]

    for _ in range(n_iter):
        old = amplitudes.copy()
        mu = np.maximum(matrix @ amplitudes + background, float(eps))
        ratio = counts / mu
        numerator = matrix.T @ ratio
        amplitudes = amplitudes * numerator / denominator
        amplitudes = np.maximum(
            np.nan_to_num(amplitudes, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )
        objective.append(poisson_objective(matrix, amplitudes, counts, background, lam))
        rel_change.append(
            float(np.linalg.norm(amplitudes - old) / (np.linalg.norm(old) + float(eps)))
        )
        projection_nrmse.append(
            float(
                np.linalg.norm((matrix @ amplitudes + background) - counts)
                / (np.linalg.norm(counts) + float(eps))
            )
        )

    logs = {
        "iteration": np.arange(n_iter + 1, dtype=np.int64),
        "objective": np.asarray(objective, dtype=np.float64),
        "relative_change": np.asarray(rel_change, dtype=np.float64),
        "projection_nrmse": np.asarray(projection_nrmse, dtype=np.float64),
    }
    return amplitudes, logs


def _write_objective_log(path: Path, logs: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "objective", "relative_change", "projection_nrmse"],
        )
        writer.writeheader()
        for idx in range(len(logs["iteration"])):
            writer.writerow(
                {
                    "iteration": int(logs["iteration"][idx]),
                    "objective": float(logs["objective"][idx]),
                    "relative_change": float(logs["relative_change"][idx]),
                    "projection_nrmse": float(logs["projection_nrmse"][idx]),
                }
            )


def _write_atom_metadata(path: Path, metadata: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)


def reconstruct_gaussian_dictionary(
    y: np.ndarray,
    projector,
    volume_shape: tuple[int, int, int],
    voxel_spacing: float | tuple[float, float, float] = 0.5,
    support_mask: np.ndarray | None = None,
    grid_stride: int | tuple[int, int, int] = 6,
    sigmas=(1.5,),
    normalize: str = "sum",
    b: float | np.ndarray | None = None,
    lambda_l1: float = 0.0,
    iterations: int = 100,
    init: str | np.ndarray = "matched",
    measured_angles=None,
    dictionary_chunk_size: int = 32,
    projection_cache_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the measured-view-only Gaussian dictionary EM reconstruction."""
    atoms, metadata = create_gaussian_atoms(
        volume_shape=volume_shape,
        voxel_spacing=voxel_spacing,
        support_mask=support_mask,
        grid_stride=grid_stride,
        sigmas=sigmas,
        normalize=normalize,
        return_metadata=True,
    )
    P = build_projection_dictionary(
        atoms=atoms,
        measured_angles=measured_angles,
        projector=projector,
        chunk_size=dictionary_chunk_size,
        cache_path=projection_cache_path,
    )
    amplitudes, logs = poisson_em_amplitude_fit(
        P=P,
        y=y,
        b=b,
        lambda_l1=lambda_l1,
        iterations=iterations,
        init=init,
    )
    reconstruction = reconstruct_from_amplitudes(atoms, amplitudes)
    prediction = P @ amplitudes + _background_vector(b, np.asarray(y).size)
    data_fit_nrmse = float(
        np.linalg.norm(prediction - np.asarray(y, dtype=np.float64).reshape(-1))
        / (np.linalg.norm(np.asarray(y, dtype=np.float64).reshape(-1)) + EPS)
    )
    threshold = max(float(np.max(amplitudes)) * 1.0e-4, EPS)
    summary = {
        "num_atoms": int(atoms.shape[0]),
        "num_nonzero_amplitudes": int(np.count_nonzero(amplitudes > threshold)),
        "lambda_l1": float(lambda_l1),
        "iterations": int(iterations),
        "final_objective": float(logs["objective"][-1]),
        "data_fit_nrmse": data_fit_nrmse,
        "atom_normalize": str(normalize),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "gaussian_dictionary_reconstruction.npy", reconstruction)
        np.save(out_dir / "amplitudes.npy", amplitudes)
        np.savez_compressed(
            out_dir / "gaussian_dictionary_result.npz",
            reconstruction=reconstruction,
            amplitudes=amplitudes,
            prediction=prediction,
            data_fit_nrmse=data_fit_nrmse,
            num_atoms=int(atoms.shape[0]),
            num_nonzero_amplitudes=summary["num_nonzero_amplitudes"],
            lambda_l1=float(lambda_l1),
            iterations=int(iterations),
        )
        _write_objective_log(out_dir / "objective_log.csv", logs)
        _write_atom_metadata(out_dir / "atom_metadata.csv", metadata)

    return {
        "reconstruction": reconstruction,
        "amplitudes": amplitudes,
        "atoms": atoms,
        "atom_metadata": metadata,
        "projection_dictionary": P,
        "prediction": prediction,
        "logs": logs,
        "summary": summary,
    }

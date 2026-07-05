"""Module B — Bayesian Optimization / Active Learning for target matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize

from model import TARGET_COLUMNS, ReactionSurrogateModel

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Recommended experimental settings."""

    recommendations: np.ndarray
    acquisition_values: np.ndarray
    optimizer_messages: list[str]
    converged: np.ndarray


def _validate_targets(targets: Mapping[str, float]) -> np.ndarray:
    missing = set(TARGET_COLUMNS) - set(targets.keys())
    if missing:
        raise ValueError(f"Missing target values for: {sorted(missing)}")
    extra = set(targets.keys()) - set(TARGET_COLUMNS)
    if extra:
        raise ValueError(f"Unexpected target properties: {sorted(extra)}")
    return np.array([targets[col] for col in TARGET_COLUMNS], dtype=float)


def target_matching_acquisition(
    X: np.ndarray,
    model: ReactionSurrogateModel,
    target_vector: np.ndarray,
    exploitation_weight: float = 1.0,
    exploration_weight: float = 0.25,
    suspension_weight: float = 1.0,
) -> float:
    """
    Custom acquisition balancing target proximity, exploration,
    and probability of achieving suspension (SIM).

    Higher values are better.
    """
    mean, std = model.predict(X.reshape(1, -1), return_std=True)
    mean = mean.ravel()
    std = np.maximum(std.ravel(), 1e-9)

    normalized_sq_error = np.sum(((mean - target_vector) / std) ** 2)
    exploitation = -normalized_sq_error
    exploration = np.sum(np.log(std))

    score = exploitation_weight * exploitation + exploration_weight * exploration

    if model.has_gpc:
        p_sim = model.predict_suspension(X.reshape(1, -1))[0]
        score += suspension_weight * np.log(max(p_sim, 1e-9))

    return score


def suggest_next_experiments(
    model: ReactionSurrogateModel,
    targets: Mapping[str, float],
    bounds: Optional[np.ndarray] = None,
    n_recommendations: int = 1,
    exploitation_weight: float = 1.0,
    exploration_weight: float = 0.25,
    suspension_weight: float = 1.0,
    n_random_restarts: int = 25,
    seed: int = 42,
) -> OptimizationResult:
    """
    Optimize the acquisition function over feature bounds.

    Parameters
    ----------
    targets:
        Mapping from continuous target name to desired value
        (Viscosidade, Escoamento).
    suspension_weight:
        Weight for log P(Suspensão = SIM) in the acquisition function.
        Higher values penalize formulations unlikely to form a suspension.
    """
    if not model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted before optimization.")

    target_vector = _validate_targets(targets)
    bounds = np.asarray(
        bounds if bounds is not None else model.feature_bounds(),
        dtype=float,
    )

    if bounds.shape != (model.n_features, 2):
        raise ValueError(
            f"Expected bounds with shape ({model.n_features}, 2), "
            f"got {bounds.shape}."
        )
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each bound must satisfy lower < upper.")

    rng = np.random.default_rng(seed)

    def objective(x: np.ndarray) -> float:
        x = np.clip(x, bounds[:, 0], bounds[:, 1])
        return target_matching_acquisition(
            x,
            model,
            target_vector,
            exploitation_weight=exploitation_weight,
            exploration_weight=exploration_weight,
            suspension_weight=suspension_weight,
        )

    candidate_points: list[np.ndarray] = []
    candidate_values: list[float] = []
    messages: list[str] = []
    converged_flags: list[bool] = []

    # Global search via differential evolution
    try:
        de_result = differential_evolution(
            lambda x: -objective(x),
            bounds=[(lo, hi) for lo, hi in bounds],
            seed=seed,
            maxiter=100,
            polish=True,
            updating="deferred",
            workers=1,
        )
        candidate_points.append(de_result.x)
        candidate_values.append(objective(de_result.x))
        converged_flags.append(bool(de_result.success))
        messages.append(de_result.message)
    except Exception as exc:
        logger.warning("Differential evolution failed: %s", exc)
        messages.append(f"Differential evolution failed: {exc}")

    # Local refinements from random restarts
    for _ in range(n_random_restarts):
        x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
        try:
            result = minimize(
                lambda x: -objective(x),
                x0=x0,
                bounds=[(lo, hi) for lo, hi in bounds],
                method="L-BFGS-B",
                options={"maxiter": 250, "ftol": 1e-9},
            )
            candidate_points.append(result.x)
            candidate_values.append(objective(result.x))
            converged_flags.append(bool(result.success))
            messages.append(result.message)
        except Exception as exc:
            logger.warning("Local optimization failed: %s", exc)
            messages.append(f"Local optimization failed: {exc}")

    if not candidate_points:
        raise RuntimeError(
            "Acquisition optimization failed to produce any candidate points."
        )

    # Select diverse top-k recommendations
    ranked_indices = np.argsort(candidate_values)[::-1]
    selected: list[int] = []
    min_distance = 0.05 * np.linalg.norm(bounds[:, 1] - bounds[:, 0])

    for idx in ranked_indices:
        point = candidate_points[idx]
        if all(
            np.linalg.norm(point - candidate_points[prev]) >= min_distance
            for prev in selected
        ):
            selected.append(idx)
        if len(selected) >= n_recommendations:
            break

    if not selected:
        selected = [ranked_indices[0]]

    recommendations = np.array([candidate_points[i] for i in selected])
    acq_values = np.array([candidate_values[i] for i in selected])
    converged = np.array([converged_flags[i] for i in selected], dtype=bool)

    if not converged.all():
        logger.warning(
            "Some acquisition optimizations did not converge: %s",
            [messages[i] for i, ok in zip(selected, converged) if not ok],
        )

    return OptimizationResult(
        recommendations=recommendations,
        acquisition_values=acq_values,
        optimizer_messages=[messages[i] for i in selected],
        converged=converged,
    )

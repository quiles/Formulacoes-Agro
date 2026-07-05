"""Module A — Data Processing & Surrogate Modeling (GPR + GPC)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Fixed column schema ────────────────────────────────────────────────────
ANNOTATION_COLUMNS = ("Material", "Característica")

# Alternate spellings accepted on CSV import
COLUMN_ALIASES = {"Caracteristica": "Característica"}

FEATURE_COLUMNS = (
    "Propilenoglicol",
    "Glicerina",
    "Polietilenoglicol",
    "Metilparabeno",
    "Sorprophor",
    "Geropon DA",
    "Antarox",
    "Rodasurf",
    "Geropon SDS",
    "H2O",
    "Imidacloprida",
    "Amido",
    "Goma xantana",
    "Alginato",
    "Carvão At",
    "Biochar",
)

TARGET_COLUMNS = ("Viscosidade", "Escoamento")

CLASSIFICATION_COLUMN = "Suspensao"

ALL_COLUMNS = (
    *ANNOTATION_COLUMNS,
    *FEATURE_COLUMNS,
    *TARGET_COLUMNS,
    CLASSIFICATION_COLUMN,
)

HIGH_VALUE_MARKER = -1

DECIMAL_PLACES = 2

NUMERIC_STORAGE_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

DEFAULT_TARGETS = {"Viscosidade": 233.88, "Escoamento": 0.2}

DEFAULT_CSV = Path(__file__).parent / "data" / "Formulacoes.csv"

DEFAULT_GPR_KERNEL = "matern"
DEFAULT_GPC_KERNEL = "rbf"

KERNEL_OPTIONS: dict[str, str] = {
    "matern": "Matérn (ν=2.5) — default GPR",
    "rbf": "RBF (Gaussian) — default GPC",
    "linear": "Linear (DotProduct)",
    "matern15": "Matérn (ν=1.5)",
    "matern05": "Matérn (ν=0.5)",
    "rational_quadratic": "Rational Quadratic",
}


def _build_base_kernel(kernel_type: str):
    """Build a base kernel (without ConstantKernel or WhiteKernel)."""
    kt = kernel_type.lower()
    if kt == "rbf":
        return RBF(length_scale=1.0)
    if kt == "matern":
        return Matern(length_scale=1.0, nu=2.5)
    if kt == "matern15":
        return Matern(length_scale=1.0, nu=1.5)
    if kt == "matern05":
        return Matern(length_scale=1.0, nu=0.5)
    if kt == "linear":
        return DotProduct(sigma_0=1.0)
    if kt == "rational_quadratic":
        return RationalQuadratic(length_scale=1.0, alpha=1.0)
    supported = ", ".join(sorted(KERNEL_OPTIONS))
    raise ValueError(f"Unsupported kernel '{kernel_type}'. Choose from: {supported}.")


def _normalize_suspension(series: pd.Series) -> pd.Series:
    """Map mixed-case Suspensão values to canonical SIM / NÃO."""
    return series.str.strip().str.upper().replace({"NAO": "NÃO"})


def _round_numeric_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Round feature and target columns to the configured decimal precision."""
    rounded = df.copy()
    for col in NUMERIC_STORAGE_COLUMNS:
        if col not in rounded.columns:
            continue
        values = pd.to_numeric(rounded[col], errors="coerce")
        mask = values.notna()
        if mask.any():
            rounded.loc[mask, col] = values.loc[mask].round(DECIMAL_PLACES)
    return rounded


class ReactionSurrogateModel:
    """
    GPR for continuous targets (Viscosidade, Escoamento) and
    GPC for binary classification (Suspensão SIM/NÃO).

    Target values equal to ``HIGH_VALUE_MARKER`` (-1) indicate
    measurements that were too high to record and are replaced by
    sentinel values (2× the maximum observed positive value) so the
    GPR can learn to avoid those regions.
    """

    def __init__(
        self,
        gpr_kernel_type: str = DEFAULT_GPR_KERNEL,
        gpc_kernel_type: str = DEFAULT_GPC_KERNEL,
        noise_level: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.gpr_kernel_type = gpr_kernel_type.lower()
        self.gpc_kernel_type = gpc_kernel_type.lower()
        _build_base_kernel(self.gpr_kernel_type)
        _build_base_kernel(self.gpc_kernel_type)
        self.noise_level = noise_level
        self.random_state = random_state

        self._full_df: Optional[pd.DataFrame] = None
        self._X_all: Optional[np.ndarray] = None
        self._y_cont: Optional[np.ndarray] = None
        self._y_susp: Optional[np.ndarray] = None
        self._gpr_mask: Optional[np.ndarray] = None

        self.feature_scaler = StandardScaler()
        self.target_scalers: list[StandardScaler] = []
        self.gpr_models: list[GaussianProcessRegressor] = []
        self.gpc_model: Optional[GaussianProcessClassifier] = None

    # ── Properties ──────────────────────────────────────────────────────
    @property
    def n_features(self) -> int:
        return len(FEATURE_COLUMNS)

    @property
    def n_targets(self) -> int:
        return len(TARGET_COLUMNS)

    @property
    def feature_column_names(self) -> tuple[str, ...]:
        return FEATURE_COLUMNS

    @property
    def target_column_names(self) -> tuple[str, ...]:
        return TARGET_COLUMNS

    @property
    def is_fitted(self) -> bool:
        return bool(self.gpr_models)

    @property
    def has_gpc(self) -> bool:
        return self.gpc_model is not None

    @property
    def n_experiments(self) -> int:
        return len(self._full_df) if self._full_df is not None else 0

    @property
    def n_gpr_valid(self) -> int:
        return int(self._gpr_mask.sum()) if self._gpr_mask is not None else 0

    # ── Data loading ────────────────────────────────────────────────────
    def load_csv(self, csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        return self._ingest(df)

    def _ingest(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=COLUMN_ALIASES)
        missing_cols = set(ALL_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {sorted(missing_cols)}")

        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df[CLASSIFICATION_COLUMN] = _normalize_suspension(
            df[CLASSIFICATION_COLUMN].astype(str)
        )
        invalid_susp = ~df[CLASSIFICATION_COLUMN].isin({"SIM", "NÃO"})
        if invalid_susp.any():
            logger.warning(
                "Dropping %d rows with invalid Suspensão values.",
                invalid_susp.sum(),
            )
            df = df[~invalid_susp].reset_index(drop=True)

        for col in TARGET_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self._full_df = df
        self._X_all = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        self._y_susp = (df[CLASSIFICATION_COLUMN] == "SIM").to_numpy(dtype=int)

        target_df = df[list(TARGET_COLUMNS)].copy()
        for col in TARGET_COLUMNS:
            valid_positive = target_df.loc[target_df[col] > 0, col]
            sentinel = (
                valid_positive.max() * 2.0 if len(valid_positive) > 0 else 1000.0
            )
            target_df.loc[target_df[col] == HIGH_VALUE_MARKER, col] = sentinel

        gpr_mask = target_df.notna().all(axis=1).to_numpy()
        self._gpr_mask = gpr_mask

        if gpr_mask.sum() > 0:
            self._y_cont = target_df.loc[gpr_mask].to_numpy(dtype=float)
        else:
            self._y_cont = np.empty((0, len(TARGET_COLUMNS)))

        logger.info(
            "Loaded %d rows (%d valid for GPR, %d for GPC).",
            len(df),
            gpr_mask.sum(),
            len(df),
        )
        return df

    # ── Kernel builders ─────────────────────────────────────────────────
    def _build_gpr_kernel(self):
        base = _build_base_kernel(self.gpr_kernel_type)
        return ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(
            noise_level=self.noise_level, noise_level_bounds=(1e-6, 1e1)
        )

    def _build_gpc_kernel(self):
        base = _build_base_kernel(self.gpc_kernel_type)
        return ConstantKernel(1.0, (1e-2, 1e2)) * base

    def set_kernels(
        self,
        gpr_kernel_type: Optional[str] = None,
        gpc_kernel_type: Optional[str] = None,
    ) -> ReactionSurrogateModel:
        """Update kernel types (validated but does not retrain)."""
        if gpr_kernel_type is not None:
            self.gpr_kernel_type = gpr_kernel_type.lower()
            _build_base_kernel(self.gpr_kernel_type)
        if gpc_kernel_type is not None:
            self.gpc_kernel_type = gpc_kernel_type.lower()
            _build_base_kernel(self.gpc_kernel_type)
        return self

    def _create_gpr(self, random_state: int) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=self._build_gpr_kernel(),
            normalize_y=False,
            n_restarts_optimizer=5,
            random_state=random_state,
        )

    def _create_gpc(self) -> GaussianProcessClassifier:
        return GaussianProcessClassifier(
            kernel=self._build_gpc_kernel(),
            n_restarts_optimizer=3,
            random_state=self.random_state,
        )

    # ── Fitting ─────────────────────────────────────────────────────────
    def fit(self) -> ReactionSurrogateModel:
        if self._X_all is None or self._gpr_mask is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        self.feature_scaler.fit(self._X_all)
        X_all_scaled = self.feature_scaler.transform(self._X_all)

        # GPR — one per continuous target
        self.target_scalers = []
        self.gpr_models = []
        n_gpr = self._gpr_mask.sum()

        if n_gpr >= 2:
            X_gpr_scaled = X_all_scaled[self._gpr_mask]
            for t_idx in range(len(TARGET_COLUMNS)):
                scaler = StandardScaler()
                y_scaled = scaler.fit_transform(
                    self._y_cont[:, t_idx : t_idx + 1]
                ).ravel()
                self.target_scalers.append(scaler)
                gpr = self._create_gpr(self.random_state + t_idx)
                gpr.fit(X_gpr_scaled, y_scaled)
                self.gpr_models.append(gpr)
            logger.info("Trained %d GPR(s) on %d samples.", len(self.gpr_models), n_gpr)
        else:
            logger.warning("Not enough valid rows to train GPR (need >= 2).")

        # GPC — binary suspension classifier
        if len(np.unique(self._y_susp)) >= 2:
            gpc = self._create_gpc()
            gpc.fit(X_all_scaled, self._y_susp)
            self.gpc_model = gpc
            logger.info("Trained GPC on %d samples.", len(self._X_all))
        else:
            logger.warning("Only one suspension class present; GPC not trained.")
            self.gpc_model = None

        return self

    # ── Predictions ─────────────────────────────────────────────────────
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict continuous targets (Viscosidade, Escoamento)."""
        if not self.gpr_models:
            raise RuntimeError("GPR models are not fitted.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}."
            )

        X_scaled = self.feature_scaler.transform(X)
        means, stds = [], []
        for model, scaler in zip(self.gpr_models, self.target_scalers):
            mu_sc, sigma_sc = model.predict(X_scaled, return_std=True)
            mu = scaler.inverse_transform(mu_sc.reshape(-1, 1)).ravel()
            sigma = sigma_sc * scaler.scale_[0]
            means.append(mu)
            stds.append(sigma)

        mean_arr = np.column_stack(means)
        if return_std:
            return mean_arr, np.column_stack(stds)
        return mean_arr, None

    def predict_suspension(self, X: np.ndarray) -> np.ndarray:
        """Return P(Suspensão = SIM) for each row in X."""
        if self.gpc_model is None:
            raise RuntimeError("GPC model is not fitted.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.feature_scaler.transform(X)
        proba = self.gpc_model.predict_proba(X_scaled)
        sim_idx = list(self.gpc_model.classes_).index(1)
        return proba[:, sim_idx]

    # ── Evaluation ──────────────────────────────────────────────────────
    @staticmethod
    def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
            return None
        return float(r2_score(y_true, y_pred))

    def _loocv_gpr(self, target_idx: int) -> np.ndarray:
        X_gpr = self._X_all[self._gpr_mask]
        y_col = self._y_cont[:, target_idx]
        loo = LeaveOneOut()
        preds = np.zeros(len(X_gpr))

        for train_i, test_i in loo.split(X_gpr):
            fs = StandardScaler()
            ts = StandardScaler()
            X_tr = fs.fit_transform(X_gpr[train_i])
            X_te = fs.transform(X_gpr[test_i])
            y_tr = ts.fit_transform(y_col[train_i].reshape(-1, 1)).ravel()
            gpr = self._create_gpr(self.random_state + target_idx)
            gpr.fit(X_tr, y_tr)
            pred = gpr.predict(X_te)
            preds[test_i] = ts.inverse_transform(pred.reshape(-1, 1)).ravel()[0]
        return preds

    def _loocv_gpc(self) -> np.ndarray:
        loo = LeaveOneOut()
        preds = np.zeros(len(self._X_all), dtype=int)

        for train_i, test_i in loo.split(self._X_all):
            if len(np.unique(self._y_susp[train_i])) < 2:
                preds[test_i] = self._y_susp[train_i][0]
                continue
            fs = StandardScaler()
            X_tr = fs.fit_transform(self._X_all[train_i])
            X_te = fs.transform(self._X_all[test_i])
            gpc = self._create_gpc()
            gpc.fit(X_tr, self._y_susp[train_i])
            preds[test_i] = gpc.predict(X_te)[0]
        return preds

    def evaluate(self) -> pd.DataFrame:
        """
        Compute per-target metrics.

        Returns a dataframe with columns:
        property, mae, r2, balanced_accuracy, evaluation.
        """
        if self._X_all is None or self._gpr_mask is None:
            raise RuntimeError("No data loaded.")

        rows: list[dict] = []
        n_gpr = self._gpr_mask.sum()

        # GPR metrics
        if self.gpr_models and n_gpr >= 3:
            for t_idx, name in enumerate(TARGET_COLUMNS):
                y_true = self._y_cont[:, t_idx]
                y_pred = self._loocv_gpr(t_idx)
                rows.append(
                    {
                        "property": name,
                        "mae": float(mean_absolute_error(y_true, y_pred)),
                        "r2": self._safe_r2(y_true, y_pred),
                        "balanced_accuracy": None,
                        "evaluation": "leave_one_out",
                    }
                )
        elif self.gpr_models and n_gpr >= 2:
            X_gpr = self._X_all[self._gpr_mask]
            y_pred_all, _ = self.predict(X_gpr)
            for t_idx, name in enumerate(TARGET_COLUMNS):
                y_true = self._y_cont[:, t_idx]
                y_pred = y_pred_all[:, t_idx]
                rows.append(
                    {
                        "property": name,
                        "mae": float(mean_absolute_error(y_true, y_pred)),
                        "r2": self._safe_r2(y_true, y_pred),
                        "balanced_accuracy": None,
                        "evaluation": "in_sample",
                    }
                )

        # GPC metrics
        if self.gpc_model is not None and len(self._X_all) >= 3:
            y_pred = self._loocv_gpc()
            bal_acc = balanced_accuracy_score(self._y_susp, y_pred)
            rows.append(
                {
                    "property": "Suspensão",
                    "mae": None,
                    "r2": None,
                    "balanced_accuracy": float(bal_acc),
                    "evaluation": "leave_one_out",
                }
            )
        elif self.gpc_model is not None:
            proba = self.predict_suspension(self._X_all)
            pred = (proba >= 0.5).astype(int)
            bal_acc = balanced_accuracy_score(self._y_susp, pred)
            rows.append(
                {
                    "property": "Suspensão",
                    "mae": None,
                    "r2": None,
                    "balanced_accuracy": float(bal_acc),
                    "evaluation": "in_sample",
                }
            )

        return pd.DataFrame(rows)

    def confusion_matrix_loocv(self) -> tuple[np.ndarray, list[str]]:
        """
        LOOCV confusion matrix for the GPC suspension classifier.

        Returns (matrix, labels) where labels = ["NÃO", "SIM"].
        """
        if self.gpc_model is None or self._y_susp is None:
            raise RuntimeError("GPC model is not fitted.")
        y_pred = self._loocv_gpc()
        cm = confusion_matrix(self._y_susp, y_pred, labels=[0, 1])
        return cm, ["NÃO", "SIM"]

    # ── Update ──────────────────────────────────────────────────────────
    def update(self, new_row: pd.DataFrame) -> ReactionSurrogateModel:
        """Append new experiment(s) and retrain all models."""
        if self._full_df is None:
            raise RuntimeError("No data loaded.")

        missing_cols = set(ALL_COLUMNS) - set(new_row.columns)
        if missing_cols:
            raise ValueError(f"New row is missing columns: {sorted(missing_cols)}")

        new_rows = new_row.loc[:, list(ALL_COLUMNS)]
        if new_rows.isna().any().any():
            raise ValueError("New rows contain missing values.")

        new_rows = _round_numeric_storage(new_rows)
        updated = pd.concat([self._full_df, new_rows], ignore_index=True)
        self._ingest(updated)
        return self.fit()

    def to_dataframe(self) -> pd.DataFrame:
        if self._full_df is None:
            raise RuntimeError("No data loaded.")
        return self._full_df.copy()

    # ── Feature bounds ──────────────────────────────────────────────────
    def feature_bounds(self, margin: float = 0.05) -> np.ndarray:
        """Non-negative box constraints inferred from observed feature ranges."""
        if self._X_all is None:
            raise RuntimeError("No data loaded.")
        lo = self._X_all.min(axis=0)
        hi = self._X_all.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        bounds = np.column_stack([lo - margin * span, hi + margin * span])
        bounds[:, 0] = np.maximum(bounds[:, 0], 0.0)
        return bounds

    # ── Sweep for plotting ──────────────────────────────────────────────
    def predict_sweep(
        self,
        feature_index: int,
        n_points: int = 200,
        reference: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sweep one feature across its range, holding others at the median.

        Returns (x_values, mean, std, p_sim) where p_sim is the GPC
        probability of Suspensão = SIM.
        """
        if self._X_all is None:
            raise RuntimeError("No data loaded.")

        if reference is None:
            reference = np.median(self._X_all, axis=0)

        bounds = self.feature_bounds(margin=0.0)
        x_vals = np.linspace(
            bounds[feature_index, 0], bounds[feature_index, 1], n_points
        )
        X_grid = np.tile(reference, (n_points, 1))
        X_grid[:, feature_index] = x_vals

        mean, std = self.predict(X_grid, return_std=True)

        if self.gpc_model is not None:
            p_sim = self.predict_suspension(X_grid)
        else:
            p_sim = np.full(n_points, np.nan)

        return x_vals, mean, std, p_sim

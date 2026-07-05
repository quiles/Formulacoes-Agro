"""Module C — Streamlit front-end for closed-loop formulation optimization."""

from __future__ import annotations

import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import (
    ANNOTATION_COLUMNS,
    CLASSIFICATION_COLUMN,
    DEFAULT_TARGETS,
    FEATURE_COLUMNS,
    HIGH_VALUE_MARKER,
    TARGET_COLUMNS,
    ReactionSurrogateModel,
)
from optimizer import suggest_next_experiments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Formulation Bayesian Optimization",
    page_icon="⚗️",
    layout="wide",
)


# ── Session state ───────────────────────────────────────────────────────
def _load_model_from_dataframe(df: pd.DataFrame) -> ReactionSurrogateModel:
    model = ReactionSurrogateModel()
    model._ingest(df)
    model.fit()
    return model


def _load_model_from_upload(uploaded_file) -> ReactionSurrogateModel:
    content = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    return _load_model_from_dataframe(df)


def _handle_csv_upload(uploaded_file) -> None:
    """Load an uploaded CSV and refresh session state."""
    upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("loaded_upload_key") == upload_key:
        return

    try:
        st.session_state.model = _load_model_from_upload(uploaded_file)
        st.session_state.loaded_upload_key = upload_key
        st.session_state.last_recommendation = None
        st.session_state.last_optimization_meta = None
        st.rerun()
    except Exception as exc:
        st.error(f"Não foi possível carregar o CSV: {exc}")


def _init_session_state() -> None:
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.loaded_upload_key = None
        try:
            model = ReactionSurrogateModel()
            model.load_csv()
            model.fit()
            st.session_state.model = model
        except FileNotFoundError:
            logger.info("Default CSV not found; waiting for user upload.")
        except Exception as exc:
            logger.warning("Unable to load default dataset: %s", exc)

    for key in ("last_recommendation", "last_optimization_meta"):
        if key not in st.session_state:
            st.session_state[key] = None


def _render_csv_uploader(*, label: str, key: str) -> None:
    uploaded = st.file_uploader(
        label,
        type=["csv"],
        key=key,
        help="Arquivo com as 21 colunas de formulações (Material, features, targets).",
    )
    if uploaded is not None:
        _handle_csv_upload(uploaded)


def _render_sidebar_dataset_controls() -> None:
    st.header("Dataset")

    if st.session_state.model is None:
        st.info(
            "Nenhum arquivo de dados encontrado. "
            "Carregue a planilha CSV abaixo para iniciar."
        )
        _render_csv_uploader(label="Carregar CSV", key="sidebar_csv_uploader")
    else:
        _render_csv_uploader(
            label="Substituir CSV",
            key="sidebar_csv_replace_uploader",
        )

    if st.session_state.model is not None:
        st.divider()
        df = st.session_state.model.to_dataframe()
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download current dataset",
            data=csv_buffer.getvalue(),
            file_name="Formulacoes_updated.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_missing_dataset_view() -> None:
    st.warning(
        "Para utilizar a aplicação, é necessário carregar a planilha de formulações."
    )
    st.markdown(
        """
        Selecione o arquivo **CSV** com os experimentos já realizados.
        Após o carregamento, os modelos serão treinados automaticamente.
        """
    )
    _render_csv_uploader(label="Carregar planilha CSV", key="main_csv_uploader")

    with st.expander("Formato esperado do CSV"):
        st.markdown(
            """
            - **Anotações:** Material, Característica
            - **Features (16):** Propilenoglicol, Glicerina, Polietilenoglicol,
              Metilparabeno, Sorprophor, Geropon DA, Antarox, Rodasurf,
              Geropon SDS, H2O, Imidacloprida, Amido, Goma xantana,
              Alginato, Carvão At, Biochar
            - **Targets:** Viscosidade, Escoamento
            - **Classificação:** Suspensao (`SIM` ou `NÃO`)
            - Valores das features ≥ 0 (zero = insumo não utilizado)
            """
        )


# ── Diagnostic plots ───────────────────────────────────────────────────
def _render_diagnostic_plots(model: ReactionSurrogateModel, df: pd.DataFrame) -> None:
    gpr_mask = model._gpr_mask

    # Parity plots (observed vs. predicted) — continuous targets only
    st.markdown("#### Observed vs. predicted")
    n_targets = model.n_targets
    fig_par, axes_par = plt.subplots(
        1, n_targets, figsize=(5 * n_targets, 4.5), squeeze=False
    )

    X_gpr = model._X_all[gpr_mask]
    y_pred_all, _ = model.predict(X_gpr)

    for t_idx, name in enumerate(TARGET_COLUMNS):
        ax = axes_par[0, t_idx]
        y_obs = model._y_cont[:, t_idx]
        y_pred = y_pred_all[:, t_idx]
        lo = min(y_obs.min(), y_pred.min())
        hi = max(y_obs.max(), y_pred.max())
        margin = (hi - lo) * 0.08

        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            ls="--", color="#888", lw=1, zorder=1,
        )
        ax.scatter(y_obs, y_pred, s=48, edgecolors="k", linewidths=0.5, zorder=2)
        ax.set_xlabel(f"Observed {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")

    fig_par.tight_layout()
    st.pyplot(fig_par)
    plt.close(fig_par)

    # Confusion matrix — GPC suspension classifier
    if model.has_gpc:
        st.markdown("#### Suspension confusion matrix (Leave-one-out CV)")
        try:
            cm, labels = model.confusion_matrix_loocv()
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            im = ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_xticks(range(len(labels)))
            ax_cm.set_yticks(range(len(labels)))
            ax_cm.set_xticklabels(labels)
            ax_cm.set_yticklabels(labels)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Observed")
            ax_cm.set_title("Suspensão")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    val = cm[i, j]
                    color = "white" if val > cm.max() / 2 else "black"
                    ax_cm.text(
                        j, i, str(val), ha="center", va="center",
                        fontsize=16, fontweight="bold", color=color,
                    )
            fig_cm.colorbar(im, ax=ax_cm, shrink=0.8)
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)
        except Exception as exc:
            st.warning(f"Unable to compute confusion matrix: {exc}")

    # Feature-effect plots
    st.markdown("#### Feature effect on targets (GPR surface slice)")
    st.caption(
        "Each curve sweeps one feature while holding the others at their "
        "median. Shaded area = ± 2σ."
    )

    for f_idx, feat_name in enumerate(FEATURE_COLUMNS):
        x_vals, mean, std, _p_sim = model.predict_sweep(f_idx)

        fig_fe, axes_fe = plt.subplots(
            1, n_targets, figsize=(5 * n_targets, 4), squeeze=False
        )

        for t_idx, target_name in enumerate(TARGET_COLUMNS):
            ax = axes_fe[0, t_idx]
            mu = mean[:, t_idx]
            sigma = std[:, t_idx]
            ax.fill_between(
                x_vals, mu - 2 * sigma, mu + 2 * sigma,
                alpha=0.18, color="tab:blue", label="± 2σ",
            )
            ax.plot(x_vals, mu, color="tab:blue", lw=1.8, label="GPR mean")

            valid_feat = df.loc[gpr_mask, feat_name].values
            valid_target = df.loc[gpr_mask, target_name].values
            ax.scatter(
                valid_feat, valid_target, s=40, color="tab:orange",
                edgecolors="k", linewidths=0.5, zorder=3, label="Observed",
            )
            ax.set_xlabel(feat_name)
            ax.set_ylabel(target_name)
            ax.set_title(f"{target_name} vs {feat_name}")
            ax.legend(fontsize=8, loc="best")

        fig_fe.tight_layout()
        st.pyplot(fig_fe)
        plt.close(fig_fe)


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session_state()

    st.title("Closed-Loop Formulation Optimization")
    st.caption(
        "Bayesian optimization over chemical formulations: GPR for Viscosidade "
        "and Escoamento, GPC for Suspensão (SIM/NÃO)."
    )

    with st.sidebar:
        _render_sidebar_dataset_controls()

    if st.session_state.model is None:
        _render_missing_dataset_view()
        return

    model: ReactionSurrogateModel = st.session_state.model
    df = model.to_dataframe()

    # ── Dataset ─────────────────────────────────────────────────────
    st.subheader("Current dataset")
    st.dataframe(df, use_container_width=True)
    st.write(
        f"{model.n_experiments} experiments · "
        f"{model.n_gpr_valid} with valid continuous targets · "
        f"{model.n_features} features"
    )

    # ── Model accuracy ──────────────────────────────────────────────
    st.subheader("Model accuracy")
    try:
        metrics_df = model.evaluate()
        disp = metrics_df.copy()
        disp["mae"] = disp["mae"].map(
            lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        )
        disp["r2"] = disp["r2"].map(
            lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        )
        disp["balanced_accuracy"] = disp["balanced_accuracy"].map(
            lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        )
        disp["evaluation"] = disp["evaluation"].map(
            {"leave_one_out": "Leave-one-out CV", "in_sample": "In-sample"}
        )
        st.dataframe(
            disp.rename(columns={
                "property": "Property",
                "mae": "MAE",
                "r2": "R²",
                "balanced_accuracy": "Balanced Acc.",
                "evaluation": "Evaluation",
            }),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.warning(f"Unable to compute model metrics: {exc}")

    # ── Diagnostic plots ────────────────────────────────────────────
    show_plots = st.toggle("Show diagnostic plots", value=False)
    if show_plots:
        _render_diagnostic_plots(model, df)

    st.divider()

    # ── Target values ───────────────────────────────────────────────
    st.subheader("Target property values")

    target_values: dict[str, float] = {}
    t_cols = st.columns(len(TARGET_COLUMNS))
    for idx, name in enumerate(TARGET_COLUMNS):
        with t_cols[idx]:
            target_values[name] = st.number_input(
                name,
                value=DEFAULT_TARGETS[name],
                format="%.4f",
                key=f"target_{name}",
            )

    st.caption("The optimizer also maximizes P(Suspensão = SIM) automatically.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        exploitation_weight = st.slider("Exploitation weight", 0.1, 5.0, 1.0, 0.1)
    with col2:
        exploration_weight = st.slider("Exploration weight", 0.0, 2.0, 0.25, 0.05)
    with col3:
        suspension_weight = st.slider("Suspension weight", 0.0, 5.0, 1.0, 0.1)
    with col4:
        n_recommendations = st.number_input(
            "Number of suggestions", min_value=1, max_value=5, value=1, step=1,
        )

    # ── Run optimization ────────────────────────────────────────────
    if st.button("Run Bayesian Optimization", type="primary"):
        try:
            result = suggest_next_experiments(
                model=model,
                targets=target_values,
                n_recommendations=int(n_recommendations),
                exploitation_weight=exploitation_weight,
                exploration_weight=exploration_weight,
                suspension_weight=suspension_weight,
            )
            st.session_state.last_recommendation = result
            st.session_state.last_optimization_meta = {
                "targets": target_values.copy(),
                "weights": {
                    "exploitation": exploitation_weight,
                    "exploration": exploration_weight,
                    "suspension": suspension_weight,
                },
            }
            if not result.converged.all():
                st.warning(
                    "Optimization finished with convergence warnings. "
                    "Review recommendations carefully."
                )
            else:
                st.success("Optimization completed successfully.")
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")

    # ── Display recommendations ─────────────────────────────────────
    if st.session_state.last_recommendation is not None:
        result = st.session_state.last_recommendation
        meta = st.session_state.last_optimization_meta or {}
        tv = meta.get("targets", target_values)

        st.subheader("Recommended experimental parameters")

        for rec_idx, (params, score, conv) in enumerate(
            zip(result.recommendations, result.acquisition_values, result.converged),
            start=1,
        ):
            st.markdown(f"**Recommendation {rec_idx}**")
            rec_df = pd.DataFrame([params], columns=list(FEATURE_COLUMNS))
            st.dataframe(rec_df, use_container_width=True)
            st.write(f"Acquisition score: `{score:.4f}` · Converged: `{conv}`")

            mean, std = model.predict(params, return_std=True)
            pred_rows = []
            for name, mu, sigma in zip(TARGET_COLUMNS, mean.ravel(), std.ravel()):
                pred_rows.append({
                    "Property": name,
                    "Predicted": f"{mu:.4f}",
                    "± σ": f"{sigma:.4f}",
                    "Target": f"{tv[name]:.4f}",
                })

            if model.has_gpc:
                p_sim = model.predict_suspension(params)[0]
                pred_rows.append({
                    "Property": "Suspensão",
                    "Predicted": f"P(SIM) = {p_sim:.4f}",
                    "± σ": "—",
                    "Target": "SIM",
                })

            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Submit lab results ──────────────────────────────────────────
    st.subheader("Submit lab results")

    if st.session_state.last_recommendation is None:
        st.info("Run Bayesian Optimization first to get a parameter suggestion.")
    else:
        latest = st.session_state.last_recommendation.recommendations[0]
        with st.form("lab_result_form"):
            st.markdown("**Annotations**")
            ann_cols = st.columns(2)
            with ann_cols[0]:
                material_val = st.text_input("Material")
            with ann_cols[1]:
                caract_val = st.text_input("Característica")

            st.markdown("**Feature values** (adjust if needed)")
            feature_inputs: dict[str, float] = {}
            feat_cols_per_row = 4
            for row_start in range(0, len(FEATURE_COLUMNS), feat_cols_per_row):
                row_feats = FEATURE_COLUMNS[row_start : row_start + feat_cols_per_row]
                cols = st.columns(len(row_feats))
                for col_ui, feat_name in zip(cols, row_feats):
                    with col_ui:
                        feat_idx = FEATURE_COLUMNS.index(feat_name)
                        feature_inputs[feat_name] = st.number_input(
                            feat_name,
                            value=float(latest[feat_idx]),
                            format="%.4f",
                            key=f"feat_{feat_name}",
                        )

            st.markdown("**Observed targets**")
            st.caption(
                f"Enter {HIGH_VALUE_MARKER} if the value was too high to measure."
            )
            obs_cols = st.columns(3)
            with obs_cols[0]:
                obs_visc = st.number_input(
                    "Viscosidade",
                    value=float(target_values.get("Viscosidade", 0)),
                    format="%.4f",
                )
            with obs_cols[1]:
                obs_esc = st.number_input(
                    "Escoamento",
                    value=float(target_values.get("Escoamento", 0)),
                    format="%.4f",
                )
            with obs_cols[2]:
                obs_susp = st.selectbox("Suspensão", ["SIM", "NÃO"])

            submitted = st.form_submit_button("Add experiment and retrain models")

        if submitted:
            try:
                new_row = {
                    "Material": material_val,
                    "Característica": caract_val,
                    **feature_inputs,
                    "Viscosidade": obs_visc,
                    "Escoamento": obs_esc,
                    CLASSIFICATION_COLUMN: obs_susp,
                }
                new_df = pd.DataFrame([new_row])
                model.update(new_df)
                st.session_state.model = model
                st.session_state.last_recommendation = None
                st.success("Experiment added and models retrained.")
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to update model: {exc}")


if __name__ == "__main__":
    main()

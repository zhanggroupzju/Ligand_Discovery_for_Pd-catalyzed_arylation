import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from joblib import dump
import os
import datetime
import warnings
import shutil

from sklearn.svm import SVR
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import random
import json

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.unicode_minus": False,
})


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


set_seed()


# ==========================
# Configuration constants
# ==========================


TRAIN_FILE_PATH = "train_data_ddG.xlsx"
TEST_FILE_PATH = "test_data_ddG.xlsx"
VAL_FILE_PATH = ""


TARGET_NAME = "ΔΔG "
TARGET_MIN = -1.0
TARGET_MAX = 2.2


CORR_THRESHOLD_CANDIDATES = [0.8, 0.81,0.82,0.83,0.84,0.85,0.87,0.88, 0.9]
PCA_N_COMPONENTS_CANDIDATES = [3, 4, 5, 6,7, 8,9, 10]


SELECTED_KERNEL = "poly"
RUN_OPTUNA = True
OPTUNA_TRIALS = 400
OPTUNA_TIMEOUT = 750


PLOT_WIDTH = 3.0
PLOT_HEIGHT = 3.0
TICK_LABEL_FONTSIZE = 7
ASPECT_RATIO = 519 / 498


USE_LOOCV = True          # True = LOOCV, False = KFold
K_FOLD_SPLITS = 5


OUTPUT_FOLDER_NAME = None


def read_data(path):
    return pd.read_excel(path)


def preprocess_data(
    data: pd.DataFrame,
    is_test: bool = False,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
    dropped_cols: set[str] | None = None,
    pca_n_components: int | None = None,
):
    """
    Preprocess data: column selection, optional correlated-feature dropping,
    scaling and PCA.
    """
    if not is_test:
        X = data.iloc[:, 2:154].copy()
        Y = data.iloc[:, 1].copy()
        original_indices = data.iloc[:, 0].copy()

        if Y.max() > TARGET_MAX or Y.min() < TARGET_MIN:
            print(
                f"Warning: training target out of expected range "
                f"(expected [{TARGET_MIN}, {TARGET_MAX}], "
                f"min={Y.min():.4f}, max={Y.max():.4f})."
            )
        else:
            print(f"Training target range: min={Y.min():.4f}, max={Y.max():.4f}")
    else:
        X = data.iloc[:, 2:154].copy()
        Y = data.iloc[:, 1].copy() if len(data.columns) > 1 else None
        original_indices = data.iloc[:, 0].copy()

        if Y is not None:
            if Y.max() > TARGET_MAX or Y.min() < TARGET_MIN:
                print(
                    f"Warning: test target out of expected range "
                    f"(expected [{TARGET_MIN}, {TARGET_MAX}], "
                    f"min={Y.min():.4f}, max={Y.max():.4f})."
                )
            else:
                print(f"Test target range: min={Y.min():.4f}, max={Y.max():.4f}")

    # Drop previously identified highly correlated features (based on training set)
    if dropped_cols is not None:
        cols_to_drop = [c for c in dropped_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

    # Drop constant columns only on the training set
    if not is_test:
        constant_cols = [c for c in X.columns if X[c].nunique() == 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)

    # Drop columns containing NaN/inf
    nan_mask = X.isna().any() | np.isinf(X).any()
    nan_cols = X.columns[nan_mask].tolist()
    if nan_cols:
        X = X.drop(columns=nan_cols)

    # Standardization
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # PCA
    if pca is not None:
        X_pca = pca.transform(X_scaled)
    else:
        if pca_n_components is None:
            raise ValueError("pca_n_components must be provided when fitting PCA.")
        max_valid_components = min(X_scaled.shape[0], X_scaled.shape[1])
        if pca_n_components > max_valid_components:
            raise ValueError(
                f"pca_n_components={pca_n_components} exceeds "
                f"max valid components={max_valid_components}."
            )
        pca = PCA(n_components=pca_n_components)
        X_pca = pca.fit_transform(X_scaled)
        print("Explained variance ratio by each PC:")
        for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
            print(f"  PC{i}: {ratio:.4f} ({ratio * 100:.2f}%)")

        print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_):.4f} "
              f"({np.sum(pca.explained_variance_ratio_) * 100:.2f}%)")

    print(f"Preprocessing finished: samples={X_pca.shape[0]}, features={X_pca.shape[1]}")
    return X_pca, Y, scaler, pca, original_indices


def remove_highly_correlated_features(
    X: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Remove highly correlated features based on a Spearman correlation threshold.
    """
    corr_matrix = X.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    for col in upper.columns:
        for row in upper.index:
            if row != col and upper.loc[row, col] > threshold:
                if col not in to_drop:
                    to_drop.add(col)

    X_clean = X.drop(columns=list(to_drop))

    print("High-correlation feature filtering (Spearman):")
    print(f"- Original features: {X.shape[1]}")
    print(f"- Correlation threshold: {threshold}")
    print(f"- Number of dropped features: {len(to_drop)}")
    print(f"- Features after filtering: {X_clean.shape[1]}")

    return X_clean, to_drop


def train_model(X_train: np.ndarray, y_train: np.ndarray, svr_params: dict) -> SVR:
    model = SVR(**svr_params)
    model.fit(X_train, y_train.ravel())
    return model


def predict_model(model: SVR, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)


def cross_val_predict_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    svr_params: dict,
    cv_strategy: str = "kfold",
    n_splits: int = 5,
) -> np.ndarray:
    if cv_strategy == "kfold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"\nStarting {n_splits}-fold CV using SVR...")
    elif cv_strategy == "loocv":
        cv = LeaveOneOut()
        print("\nStarting Leave-One-Out CV using SVR...")
    else:
        raise ValueError("cv_strategy must be 'kfold' or 'loocv'")

    y_pred = np.zeros_like(y, dtype=float)
    total_splits = cv.get_n_splits(X)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        model = train_model(X_train, y_train, svr_params)
        pred = predict_model(model, X_test)
        y_pred[test_idx] = pred

        if (
            cv_strategy == "kfold"
            or (fold_idx + 1)
            % (total_splits // 5 if total_splits > 5 else 1)
            == 0
            or (fold_idx + 1)
            == total_splits
        ):
            print(f"Completed {fold_idx + 1}/{total_splits} folds")

    return y_pred


def objective_with_cv_simplified(
    trial: optuna.trial.Trial,
    train_data: pd.DataFrame,
    selected_kernel: str,
    use_loocv: bool = False,
    k_fold_splits: int = 5,
) -> float:

    corr_threshold = trial.suggest_categorical(
        "corr_threshold",
        CORR_THRESHOLD_CANDIDATES,
    )
    n_components = trial.suggest_categorical(
        "n_components",
        PCA_N_COMPONENTS_CANDIDATES,
    )

    C = trial.suggest_float("C", 1e-2, 1e3, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)

    svr_params: dict = {"C": C, "epsilon": epsilon, "kernel": selected_kernel}

    if selected_kernel == "rbf":
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        svr_params["gamma"] = gamma
    elif selected_kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        svr_params["degree"] = degree
        svr_params["gamma"] = gamma
    elif selected_kernel == "sigmoid":
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        svr_params["gamma"] = gamma

    print(
        f"\n  Optuna trial {trial.number}: "
        f"corr_threshold={corr_threshold}, n_components={n_components}"
    )


    X_raw = train_data.iloc[:, 2:154].copy()
    _, to_drop = remove_highly_correlated_features(X_raw, corr_threshold)

    X_pca, Y, _, _, _ = preprocess_data(
        train_data,
        pca_n_components=n_components,
        dropped_cols=to_drop,
    )
    y = Y.values

    if use_loocv:
        cv = LeaveOneOut()
        cv_name = "LOOCV"
    else:
        n_splits_optuna = min(k_fold_splits, len(y))
        if n_splits_optuna < 2:
            raise ValueError("Not enough samples for KFold during Optuna optimization.")
        cv = KFold(n_splits=n_splits_optuna, shuffle=True, random_state=42)
        cv_name = f"{n_splits_optuna}-fold KFold"

    print(f"  Optuna trial {trial.number}: using {cv_name}")

    all_y_true: list[float] = []
    all_y_pred: list[float] = []
    total_splits = cv.get_n_splits(X_pca)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_pca), start=1):
        X_train, y_train = X_pca[train_idx], y[train_idx]
        X_val, y_val = X_pca[val_idx], y[val_idx]

        model = train_model(X_train, y_train, svr_params)
        y_val_pred = predict_model(model, X_val)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_val_pred)

        current_mae = mean_absolute_error(all_y_true, all_y_pred)
        trial.report(current_mae, step=fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if (
            not use_loocv
            or fold % (total_splits // 5 if total_splits > 5 else 1) == 0
            or fold == total_splits
        ):
            print(f"    Trial {trial.number}: completed {fold}/{total_splits} folds")

    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    return overall_mae


def optimize_with_optuna_simplified(
    train_data: pd.DataFrame,
    selected_kernel: str,
    use_loocv: bool = False,
    k_fold_splits: int = 5,
    n_trials: int = 15,
    timeout: int = 1200,
):
    cv_desc = "LOOCV" if use_loocv else f"{k_fold_splits}-fold KFold"
    print(
        f"\nOptimizing SVR hyperparameters with kernel='{selected_kernel}', "
        f"CV={cv_desc}, trials={n_trials}, timeout={timeout}s"
    )
    print(f"Searching corr_threshold in {CORR_THRESHOLD_CANDIDATES}")
    print(f"Searching n_components in {PCA_N_COMPONENTS_CANDIDATES}")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective_with_cv_simplified(
            trial,
            train_data,
            selected_kernel,
            use_loocv=use_loocv,
            k_fold_splits=k_fold_splits,
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    print("\nHyperparameter optimization finished.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best MAE: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")

    best_params = study.best_params.copy()

    best_corr_threshold = best_params.pop("corr_threshold")
    best_n_components = best_params.pop("n_components")

    best_svr_params = best_params.copy()
    best_svr_params["kernel"] = selected_kernel

    return study, best_svr_params, best_corr_threshold, best_n_components


def evaluate_model(
    model_params: dict,
    X: np.ndarray,
    y: np.ndarray | None,
    data_type: str = "Training Set",
    use_cv: bool = True,
    cv_strategy: str = "kfold",
    n_splits: int = 5,
    original_indices=None,
    pre_trained_model: SVR | None = None,
    plot_figsize: tuple[int, int] = (7, 7),
    tick_label_fontsize: int = 10,
    aspect_ratio: float | None = None,
):
    if y is None:
        print(f"Warning: no target provided for {data_type}; skip metrics/plot.")
        return None, None, None, None, plt.figure(figsize=plot_figsize), None, None

    if pre_trained_model is not None:
        print(f"Evaluating {data_type} using pre-trained model...")
        y_pred = predict_model(pre_trained_model, X)
    elif use_cv:
        print(
            f"Evaluating {data_type} with "
            f"{'LOOCV' if cv_strategy == 'loocv' else f'{n_splits}-fold CV'}..."
        )
        y_pred = cross_val_predict_sklearn(
            X,
            y,
            svr_params=model_params,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
        )
    else:
        print(f"Evaluating {data_type} with a direct fit on full data...")
        temp_model = train_model(X, y, model_params)
        y_pred = predict_model(temp_model, X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = 100 * np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-8)))

    print(f"\n{data_type} metrics:")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    fig, ax = plt.subplots(1, 1, figsize=plot_figsize)

    y_plot = y
    y_pred_plot = y_pred

    min_val, max_val = TARGET_MIN, TARGET_MAX
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ticks = np.linspace(min_val, max_val, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if aspect_ratio is not None:
        ax.set_aspect(aspect_ratio)

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=0.8,
        color="#cfcfcf",
        alpha=0.7,
        zorder=1,
    )

    ax.scatter(y_plot, y_pred_plot, alpha=0.6, s=15)

    if len(y_plot) > 1 and np.std(y_plot) > 1e-6:
        slope, intercept = np.polyfit(y_plot, y_pred_plot, 1)

        x_start = max(min_val, float(np.min(y_plot)))
        x_end = min(max_val, float(np.max(y_plot)))
        x_line = np.array([x_start, x_end])
        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            color="black",
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
            zorder=3,
        )

    ax.set_title(f"{data_type} prediction", fontsize=14)
    ax.set_xlabel(f"Measured {TARGET_NAME}", fontsize=7, fontweight="bold")
    ax.set_ylabel(f"Predicted {TARGET_NAME}", fontsize=7, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)

    if data_type == "Test Set" and original_indices is not None:
        for i, txt in enumerate(original_indices):
            ax.annotate(
                str(txt),
                (y_plot[i], y_pred_plot[i]),
                textcoords="offset points",
                xytext=(5, -5),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    return r2, mae, rmse, mape, fig, model_params, y_pred


def plot_combined_predictions(
    y_train_actual: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_actual: np.ndarray | None = None,
    y_test_pred: np.ndarray | None = None,
    y_val_actual: np.ndarray | None = None,
    y_val_pred: np.ndarray | None = None,
    train_metrics_text: str | None = None,
    test_metrics_text: str | None = None,
    val_metrics_text: str | None = None,
    plot_figsize: tuple[int, int] = (7, 7),
    tick_label_fontsize: int = 10,
    aspect_ratio: float | None = None,
):

    fig, ax = plt.subplots(1, 1, figsize=plot_figsize)

    min_val, max_val = TARGET_MIN, TARGET_MAX
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ticks = np.linspace(min_val, max_val, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if aspect_ratio is not None:
        ax.set_aspect(aspect_ratio)

    # 1:1 line
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=0.8,
        color="#cfcfcf",
        alpha=0.7,
        zorder=1,
        label="1:1 Line",
    )

    # Training points
    ax.scatter(
        y_train_actual,
        y_train_pred,
        alpha=0.5,
        color="black",
        label="Training Set (LOOCV)" if USE_LOOCV else "Training (CV)",
        s=10,
        zorder=2,
    )

    # Test points (optional)
    test_color = "#0080be"
    if y_test_actual is not None and y_test_pred is not None:
        ax.scatter(
            y_test_actual,
            y_test_pred,
            alpha=0.7,
            color=test_color,
            label="Test Set",
            s=10,
            zorder=2,
        )

        # Test fit line (optional)
        if len(y_test_actual) > 1 and np.std(y_test_actual) > 1e-6:
            slope, intercept = np.polyfit(y_test_actual, y_test_pred, 1)
            x_start = max(min_val, float(np.min(y_test_actual)))
            x_end = min(max_val, float(np.max(y_test_actual)))
            x_line = np.array([x_start, x_end])
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=test_color,
                linestyle="-",
                linewidth=1.5,
                label="Test Fit",
                zorder=3,
            )

    # Validation points (optional)
    val_color = "#d0024a"
    if y_val_actual is not None and y_val_pred is not None:
        ax.scatter(
            y_val_actual,
            y_val_pred,
            alpha=0.7,
            color=val_color,
            marker="^",
            label="External Validation Set",
            s=18,
            zorder=2,
        )

        # # Validation fit line (optional)
        # if len(y_val_actual) > 1 and np.std(y_val_actual) > 1e-6:
        #     slope, intercept = np.polyfit(y_val_actual, y_val_pred, 1)
        #     x_start = max(min_val, float(np.min(y_val_actual)))
        #     x_end = min(max_val, float(np.max(y_val_actual)))
        #     x_line = np.array([x_start, x_end])
        #     y_line = slope * x_line + intercept
        #     ax.plot(
        #         x_line,
        #         y_line,
        #         color=val_color,
        #         linestyle="-",
        #         linewidth=1.2,
        #         label="Validation Fit",
        #         zorder=3,
        #     )


    ax.set_xlabel(f"Measured {TARGET_NAME} [kcal/mol]", fontsize=7, fontweight="bold")
    ax.set_ylabel(f"Predicted {TARGET_NAME} [kcal/mol]", fontsize=7, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)

    # legend: hide 1:1 & fit lines
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles, filtered_labels = [], []
    for h, lab in zip(handles, labels):
        if lab not in ("1:1 Line", "Test Fit", "Validation Fit"):
            filtered_handles.append(h)
            filtered_labels.append(lab)
    if filtered_handles:
        ax.legend(filtered_handles, filtered_labels, loc="upper left", fontsize=7, frameon=False)


    metrics_lines = []
    if train_metrics_text:
        metrics_lines.append(train_metrics_text)
    if test_metrics_text:
        metrics_lines.append(test_metrics_text)
    if val_metrics_text:
        metrics_lines.append(val_metrics_text)

    if metrics_lines:
        metrics_text = "\n".join(metrics_lines)
        ax.text(
            0.98,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.85,
            ),
            zorder=10,
        )

    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
    plt.tight_layout()
    return fig


def save_results(
    output_dir: str,
    train_metrics,
    best_params: dict,
    test_metrics=None,
    test_predictions=None,
    test_actual=None,
    val_metrics=None,
    val_predictions=None,
    val_actual=None,
    train_predictions=None,
    train_actual=None,
    train_predictions_direct=None,
    train_metrics_direct_plot=None,
    combined_plot_fig=None,
    train_sample_names=None,
    test_sample_names=None,
    val_sample_names=None,
):
    os.makedirs(output_dir, exist_ok=True)

    train_r2, train_mae, train_rmse, train_mape, train_fig_cv, _, _ = train_metrics

    if test_metrics is not None:
        test_r2, test_mae, test_rmse, test_mape, test_fig, _, _ = test_metrics
    else:
        test_r2 = test_mae = test_rmse = test_mape = None
        test_fig = None

    if val_metrics is not None:
        val_r2, val_mae, val_rmse, val_mape, val_fig, _, _ = val_metrics
    else:
        val_r2 = val_mae = val_rmse = val_mape = None
        val_fig = None

    # Figures
    train_fig_cv.savefig(
        os.path.join(output_dir, "train_performance_cv.png"),
        dpi=300,
        bbox_inches="tight",
    )

    if train_metrics_direct_plot is not None:
        train_metrics_direct_plot.savefig(
            os.path.join(output_dir, "train_performance_direct.png"),
            dpi=300,
            bbox_inches="tight",
        )

    if test_fig is not None:
        test_fig.savefig(
            os.path.join(output_dir, "test_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )

    if val_fig is not None:
        val_fig.savefig(
            os.path.join(output_dir, "validation_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )

    if combined_plot_fig is not None:
        combined_plot_fig.savefig(
            os.path.join(output_dir, "combined_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )

    # Metrics summary + tables
    with open(os.path.join(output_dir, "metrics.txt"), "w", encoding="utf-8") as f_metrics:
        f_metrics.write("=" * 60 + "\n")
        f_metrics.write("Model Evaluation Metrics\n")
        f_metrics.write("=" * 60 + "\n\n")

        f_metrics.write("Training (cross-validation):\n")
        f_metrics.write(f"  R²   : {train_r2:.4f}\n")
        f_metrics.write(f"  MAE  : {train_mae:.4f}\n")
        f_metrics.write(f"  RMSE : {train_rmse:.4f}\n")
        f_metrics.write(f"  MAPE : {train_mape:.2f}%\n\n")

        f_metrics.write("Best model parameters:\n")
        for k, v in best_params.items():
            f_metrics.write(f"  {k}: {v}\n")

        if test_metrics is not None:
            f_metrics.write("\nTest set:\n")
            f_metrics.write(f"  R²   : {test_r2:.4f}\n")
            f_metrics.write(f"  MAE  : {test_mae:.4f}\n")
            f_metrics.write(f"  RMSE : {test_rmse:.4f}\n")
            f_metrics.write(f"  MAPE : {test_mape:.2f}%\n")

        if val_metrics is not None:
            f_metrics.write("\nValidation set:\n")
            f_metrics.write(f"  R²   : {val_r2:.4f}\n")
            f_metrics.write(f"  MAE  : {val_mae:.4f}\n")
            f_metrics.write(f"  RMSE : {val_rmse:.4f}\n")
            f_metrics.write(f"  MAPE : {val_mape:.2f}%\n")

    # Prediction tables
    if train_predictions is not None and train_actual is not None:
        train_df = pd.DataFrame(
            {
                "Sample Name": train_sample_names if train_sample_names is not None else np.arange(len(train_actual)),
                "Actual": train_actual,
                "Predicted": train_predictions,
                "Error": train_actual - train_predictions,
                "Absolute Error": np.abs(train_actual - train_predictions),
                "Relative Error(%)": 100
                * np.abs((train_actual - train_predictions) / np.maximum(np.abs(train_actual), 1e-8)),
            }
        )
        train_df.to_csv(os.path.join(output_dir, "train_predictions_cv.csv"), index=False, encoding="utf-8-sig")
        train_df.to_excel(os.path.join(output_dir, "train_predictions_cv.xlsx"), index=False)

    if train_predictions_direct is not None and train_actual is not None:
        train_direct_df = pd.DataFrame(
            {
                "Sample Name": train_sample_names if train_sample_names is not None else np.arange(len(train_actual)),
                "Actual": train_actual,
                "Predicted": train_predictions_direct,
                "Error": train_actual - train_predictions_direct,
                "Absolute Error": np.abs(train_actual - train_predictions_direct),
                "Relative Error(%)": 100
                * np.abs((train_actual - train_predictions_direct) / np.maximum(np.abs(train_actual), 1e-8)),
            }
        )
        train_direct_df.to_csv(os.path.join(output_dir, "train_predictions_direct.csv"), index=False, encoding="utf-8-sig")
        train_direct_df.to_excel(os.path.join(output_dir, "train_predictions_direct.xlsx"), index=False)

    if test_predictions is not None and test_actual is not None:
        test_df = pd.DataFrame(
            {
                "Sample Name": test_sample_names if test_sample_names is not None else np.arange(len(test_actual)),
                "Actual": test_actual,
                "Predicted": test_predictions,
                "Error": test_actual - test_predictions,
                "Absolute Error": np.abs(test_actual - test_predictions),
                "Relative Error(%)": 100
                * np.abs((test_actual - test_predictions) / np.maximum(np.abs(test_actual), 1e-8)),
            }
        )
        test_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")
        test_df.to_excel(os.path.join(output_dir, "test_predictions.xlsx"), index=False)

    if val_predictions is not None and val_actual is not None:
        val_df = pd.DataFrame(
            {
                "Sample Name": val_sample_names if val_sample_names is not None else np.arange(len(val_actual)),
                "Actual": val_actual,
                "Predicted": val_predictions,
                "Error": val_actual - val_predictions,
                "Absolute Error": np.abs(val_actual - val_predictions),
                "Relative Error(%)": 100
                * np.abs((val_actual - val_predictions) / np.maximum(np.abs(val_actual), 1e-8)),
            }
        )
        val_df.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False, encoding="utf-8-sig")
        val_df.to_excel(os.path.join(output_dir, "validation_predictions.xlsx"), index=False)

    print(f"\nResults saved to folder: {output_dir}")
    return best_params


def main():
    code_file = __file__ if "__file__" in globals() else None

    train_data = read_data(TRAIN_FILE_PATH)

    kernels = ["linear", "rbf", "poly", "sigmoid"]
    selected_kernel = SELECTED_KERNEL.lower()
    if selected_kernel not in kernels:
        raise ValueError(f"SELECTED_KERNEL must be one of {kernels}, got {SELECTED_KERNEL!r}")

    print(f"\nUsing SVR kernel (from config): {selected_kernel}")

    if not RUN_OPTUNA:
        print("RUN_OPTUNA=False, skip hyperparameter search, exiting.")
        return

    if USE_LOOCV:
        print("\nOptuna CV strategy: Leave-One-Out CV (from config)")
    else:
        print(f"\nOptuna CV strategy: {K_FOLD_SPLITS}-fold KFold (from config)")

    study, best_params, best_corr_threshold, best_n_components = optimize_with_optuna_simplified(
        train_data,
        selected_kernel=selected_kernel,
        use_loocv=USE_LOOCV,
        k_fold_splits=K_FOLD_SPLITS,
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
    )

    print("\nBest preprocessing parameters found by Optuna:")
    print(f"  corr_threshold = {best_corr_threshold}")
    print(f"  n_components   = {best_n_components}")

    print("\nRunning high-correlation feature filtering with best threshold...")
    X_raw = train_data.iloc[:, 2:154]
    _, to_drop = remove_highly_correlated_features(X_raw, best_corr_threshold)

    print("\nRunning preprocessing (scaling + PCA) with best n_components...")
    X_pca, Y, scaler, pca, train_original_indices = preprocess_data(
        train_data,
        pca_n_components=best_n_components,
        dropped_cols=to_drop,
    )

    print("\nModel input summary:")
    print(f"  Features: samples={X_pca.shape[0]}, dims={X_pca.shape[1]}")
    print(f"  Targets : samples={Y.shape[0]}")

    print("\nTraining final SVR model...")
    trained_model = train_model(X_pca, Y.values, best_params)

    plot_figsize = (PLOT_WIDTH, PLOT_HEIGHT)
    tick_label_fontsize = TICK_LABEL_FONTSIZE
    aspect_ratio = ASPECT_RATIO
    print(f"Using aspect ratio: {aspect_ratio:.3f}")

    print("\nEvaluate final model generalization:")
    if USE_LOOCV:
        cv_strategy_eval = "loocv"
        n_splits_eval = None
        print("CV strategy: Leave-One-Out CV (from config)")
    else:
        cv_strategy_eval = "kfold"
        n_splits_eval = K_FOLD_SPLITS
        print(f"CV strategy: {n_splits_eval}-fold KFold (from config)")

    train_r2_cv, train_mae_cv, train_rmse_cv, train_mape_cv, fig_cv, _, train_pred_cv = evaluate_model(
        best_params,
        X_pca,
        Y.values,
        data_type="Training Set",
        use_cv=True,
        cv_strategy=cv_strategy_eval,
        n_splits=n_splits_eval if n_splits_eval is not None else 5,
        plot_figsize=plot_figsize,
        tick_label_fontsize=tick_label_fontsize,
        aspect_ratio=aspect_ratio,
    )

    train_metrics = (
        train_r2_cv,
        train_mae_cv,
        train_rmse_cv,
        train_mape_cv,
        fig_cv,
        best_params,
        train_pred_cv,
    )

    print("\nEvaluating direct fit on training set...")
    train_r2_direct, train_mae_direct, train_rmse_direct, train_mape_direct, fig_direct, _, train_pred_direct = evaluate_model(
        best_params,
        X_pca,
        Y.values,
        data_type="Training Set (Direct Fit)",
        use_cv=False,
        plot_figsize=plot_figsize,
        tick_label_fontsize=tick_label_fontsize,
        aspect_ratio=aspect_ratio,
    )

    print(
        f"Direct fit metrics: R²={train_r2_direct:.4f}, MAE={train_mae_direct:.4f}, "
        f"RMSE={train_rmse_direct:.4f}, MAPE={train_mape_direct:.2f}%"
    )


    test_file = TEST_FILE_PATH or ""
    test_metrics = None
    test_predictions = None
    test_actual = None
    test_original_indices = None

    if test_file:
        try:
            test_data = read_data(test_file)
            print("\nProcessing test data...")
            test_X_pca, test_Y, _, _, test_original_indices = preprocess_data(
                test_data,
                is_test=True,
                scaler=scaler,
                pca=pca,
                dropped_cols=to_drop,
                pca_n_components=best_n_components,
            )

            print("\nEvaluating test set (using trained SVR)...")
            test_r2, test_mae, test_rmse, test_mape, test_fig, _, test_predictions = evaluate_model(
                best_params,
                test_X_pca,
                test_Y.values if test_Y is not None else None,
                data_type="Test Set",
                use_cv=False,
                pre_trained_model=trained_model,
                original_indices=test_original_indices.values,
                plot_figsize=plot_figsize,
                tick_label_fontsize=tick_label_fontsize,
                aspect_ratio=aspect_ratio,
            )

            test_actual = test_Y.values if test_Y is not None else None
            test_metrics = (
                test_r2,
                test_mae,
                test_rmse,
                test_mape,
                test_fig,
                best_params,
                test_predictions,
            )
        except Exception as exc:
            print(f"Error while processing test data: {exc}")
            test_metrics = None
            test_predictions = None
            test_actual = None
            test_original_indices = None


    val_metrics = None
    val_predictions = None
    val_actual = None
    val_original_indices = None

    try:
        val_file_input = input("\nEnter validation file name/path (blank to skip): ").strip()
    except Exception:
        val_file_input = ""

    val_file = val_file_input or VAL_FILE_PATH or ""
    if val_file:
        try:
            val_data = read_data(val_file)
            print("\nProcessing validation data...")
            val_X_pca, val_Y, _, _, val_original_indices = preprocess_data(
                val_data,
                is_test=True,
                scaler=scaler,
                pca=pca,
                dropped_cols=to_drop,
                pca_n_components=best_n_components,
            )

            print("\nEvaluating validation set (using trained SVR)...")
            val_r2, val_mae, val_rmse, val_mape, val_fig, _, val_predictions = evaluate_model(
                best_params,
                val_X_pca,
                val_Y.values if val_Y is not None else None,
                data_type="Validation Set",
                use_cv=False,
                pre_trained_model=trained_model,
                original_indices=val_original_indices.values,
                plot_figsize=plot_figsize,
                tick_label_fontsize=tick_label_fontsize,
                aspect_ratio=aspect_ratio,
            )

            val_actual = val_Y.values if val_Y is not None else None
            val_metrics = (
                val_r2,
                val_mae,
                val_rmse,
                val_mape,
                val_fig,
                best_params,
                val_predictions,
            )
        except Exception as exc:
            print(f"Error while processing validation data: {exc}")
            val_metrics = None
            val_predictions = None
            val_actual = None
            val_original_indices = None

    folder_name = OUTPUT_FOLDER_NAME or ""
    if not folder_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        cv_label = "LOOCV" if cv_strategy_eval == "loocv" else f"{n_splits_eval}KFold"
        folder_name = f"{timestamp}_SVR_{selected_kernel.upper()}_R2_{train_r2_cv:.3f}_{cv_label}"

    output_dir = folder_name
    os.makedirs(output_dir, exist_ok=True)



    combined_plot_fig = plot_combined_predictions(
        Y.values,
        train_pred_cv,
        test_actual,
        test_predictions,
        val_actual,
        val_predictions,
        # train_metrics_text=train_metrics_text,
        # test_metrics_text=test_metrics_text,
        # val_metrics_text=val_metrics_text,
        plot_figsize=plot_figsize,
        tick_label_fontsize=tick_label_fontsize,
        aspect_ratio=aspect_ratio,
    )

    save_results(
        output_dir,
        train_metrics,
        best_params,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
        test_actual=test_actual,
        val_metrics=val_metrics,
        val_predictions=val_predictions,
        val_actual=val_actual,
        train_predictions=train_pred_cv,
        train_actual=Y.values,
        train_predictions_direct=train_pred_direct,
        train_metrics_direct_plot=fig_direct,
        combined_plot_fig=combined_plot_fig,
        train_sample_names=train_original_indices.values,
        test_sample_names=test_original_indices.values if test_original_indices is not None else None,
        val_sample_names=val_original_indices.values if val_original_indices is not None else None,
    )

    dump(study, os.path.join(output_dir, "optuna_study.joblib"))
    dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    dump(pca, os.path.join(output_dir, "pca.joblib"))
    dump(trained_model, os.path.join(output_dir, "trained_svr_model.joblib"))
    dump(to_drop, os.path.join(output_dir, "dropped_columns.joblib"))

    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        hyperparams = {
            "model_type": "SVR",
            "SVR_selected_kernel": selected_kernel,
            "SVR_best_params": best_params,
            "best_corr_threshold": best_corr_threshold,
            "best_n_components": best_n_components,
            "corr_threshold_candidates": CORR_THRESHOLD_CANDIDATES,
            "pca_n_components_candidates": PCA_N_COMPONENTS_CANDIDATES,
            "cv_strategy_for_optuna": ("loocv" if USE_LOOCV else "kfold"),
            "n_splits_for_kfold_optuna": (K_FOLD_SPLITS if not USE_LOOCV else "N/A"),
            "cv_strategy_for_evaluation": cv_strategy_eval,
            "n_splits_for_kfold_evaluation": (n_splits_eval if cv_strategy_eval == "kfold" else "N/A"),
            "plot_figsize": plot_figsize,
            "tick_label_fontsize": tick_label_fontsize,
            "plot_aspect_ratio": aspect_ratio,
            "target_name": TARGET_NAME,
            "target_min": TARGET_MIN,
            "target_max": TARGET_MAX,
        }
        json.dump(hyperparams, f, indent=2)

    if code_file:
        shutil.copy2(code_file, os.path.join(output_dir, "code_copy.py"))

    print("\nTraining and evaluation completed.")
    print("Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
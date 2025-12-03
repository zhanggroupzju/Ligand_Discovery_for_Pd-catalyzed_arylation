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

# 1. 特征处理
CORR_THRESHOLD = 0.9     # 相关性阈值
PCA_VARIANCE = 0.8       # PCA 保留累计贡献率

# 2. SVR / Optuna
SELECTED_KERNEL = "poly"  # "linear" / "rbf" / "poly" / "sigmoid"
RUN_OPTUNA = True        # 是否进行 Optuna 搜索
OPTUNA_TRIALS = 200      # Optuna trial 数
OPTUNA_TIMEOUT = 750     # Optuna 超时时间（秒）

# 3. 作图
PLOT_WIDTH = 3.0
PLOT_HEIGHT = 3.0
TICK_LABEL_FONTSIZE = 7
ASPECT_RATIO = 519 / 498  # 使用原默认纵横比

# 4. 交叉验证策略
USE_LOOCV = True         # True = LOOCV, False = KFold
K_FOLD_SPLITS = 5        # 只有当 USE_LOOCV=False 时才会用到

# 5. 数据路径 & 输出
TEST_FILE_PATH = "test_data.xlsx"   # 测试集文件路径（为空字符串则跳过）
VAL_FILE_PATH = "val_data.xlsx"     # 验证集文件路径（为空字符串则跳过）
OUTPUT_FOLDER_NAME = None           # None 或 "" 时自动生成文件夹名


def read_data(path):
    return pd.read_excel(path)


def preprocess_data(
    data: pd.DataFrame,
    is_test: bool = False,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
    dropped_cols: set[str] | None = None,
    pca_variance: float | None = None,
):
    """
    Preprocess data: column selection, optional correlated-feature dropping,
    scaling and PCA.

    Returns
    -------
    X_pca : np.ndarray
        PCA-transformed feature matrix.
    Y : pd.Series | None
        Target values (None for pure feature-only test data).
    scaler : StandardScaler
        Fitted scaler.
    pca : PCA
        Fitted PCA transformer.
    original_indices : pd.Series
        Original index/ID column from the input data.
    """
    if not is_test:
        # Keep the original column slicing consistent
        X = data.iloc[:, 2:154].copy()
        Y = data.iloc[:, 1].copy()
        original_indices = data.iloc[:, 0].copy()

        if Y.max() > 1.05 or Y.min() < 0.0:
            print(
                f"Warning: training target out of expected range "
                f"(min={Y.min():.4f}, max={Y.max():.4f})."
            )
        else:
            print(
                f"Training target range: min={Y.min():.4f}, max={Y.max():.4f}"
            )
    else:
        X = data.iloc[:, 2:154].copy()
        Y = data.iloc[:, 1].copy() if len(data.columns) > 1 else None
        original_indices = data.iloc[:, 0].copy()

        if Y is not None:
            if Y.max() > 1.05 or Y.min() < 0.0:
                print(
                    f"Warning: test/validation target out of expected range "
                    f"(min={Y.min():.4f}, max={Y.max():.4f})."
                )
            else:
                print(
                    f"Test/validation target range: min={Y.min():.4f}, max={Y.max():.4f}"
                )

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
        if pca_variance is None:
            raise ValueError("pca_variance must be provided when fitting PCA.")
        pca = PCA(n_components=pca_variance)
        X_pca = pca.fit_transform(X_scaled)

    print(f"Preprocessing finished: samples={X_pca.shape[0]}, features={X_pca.shape[1]}")
    return X_pca, Y, scaler, pca, original_indices


def remove_highly_correlated_features(
    X: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Remove highly correlated features based on a Spearman correlation threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    threshold : float
        Absolute Spearman correlation threshold above which one of the feature
        pair will be dropped.

    Returns
    -------
    X_clean : pd.DataFrame
        Feature matrix after dropping highly correlated columns.
    to_drop : set[str]
        Set of dropped column names.
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


def objective_with_kfold_simplified(
    trial: optuna.trial.Trial,
    X_pca: np.ndarray,
    y: np.ndarray,
    selected_kernel: str,
) -> float:
    C = trial.suggest_loguniform("C", 1e-2, 1e3)
    epsilon = trial.suggest_loguniform("epsilon", 1e-3, 1.0)

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

    n_splits_optuna = min(5, max(3, len(y) // 4))
    kf = KFold(n_splits=n_splits_optuna, shuffle=True, random_state=42)

    all_y_true: list[float] = []
    all_y_pred: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_pca), start=1):
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

    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    return overall_mae


def optimize_with_optuna_simplified(
    X_pca: np.ndarray,
    y: np.ndarray,
    selected_kernel: str,
    n_trials: int = 15,
    timeout: int = 1200,
):
    print(
        f"\nOptimizing SVR hyperparameters with kernel='{selected_kernel}', "
        f"trials={n_trials}, timeout={timeout}s"
    )

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective_with_kfold_simplified(
            trial, X_pca, y, selected_kernel
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    print("\nHyperparameter optimization finished.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best MAE: {study.best_value:.6f}")
    print(f"Best SVR parameters: {study.best_params}")

    best_params = study.best_params.copy()
    best_params["kernel"] = selected_kernel
    return study, best_params


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
    mape = 100 * np.mean(
        np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-8))
    )

    print(f"\n{data_type} metrics:")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    fig, ax = plt.subplots(1, 1, figsize=plot_figsize)

    y_plot = y * 100.0
    y_pred_plot = y_pred * 100.0

    min_val, max_val = 20, 100
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(np.arange(20, 101, 20))
    ax.set_yticks(np.arange(20, 101, 20))

    if aspect_ratio is not None:
        ax.set_aspect(aspect_ratio)

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.scatter(y_plot, y_pred_plot, alpha=0.6, s=15)

    ax.set_title(f"{data_type} prediction", fontsize=14)
    ax.set_xlabel("Measured R", fontsize=7, fontweight="bold")
    ax.set_ylabel("Predicted R", fontsize=7, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)

    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)

    if data_type in ("Test Set", "Validation Set") and original_indices is not None:
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
    train_r2: float | None = None,
    train_mae: float | None = None,
    test_r2: float | None = None,
    test_mae: float | None = None,
    val_r2: float | None = None,
    val_mae: float | None = None,
    plot_figsize: tuple[int, int] = (7, 7),
    tick_label_fontsize: int = 10,
    aspect_ratio: float | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=plot_figsize)

    min_val, max_val = 20, 100
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(np.arange(20, 101, 20))
    ax.set_yticks(np.arange(20, 101, 20))

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
        label="1:1 Line",
    )

    train_actual_plot = y_train_actual * 100.0
    train_pred_plot = y_train_pred * 100.0

    ax.scatter(
        train_actual_plot,
        train_pred_plot,
        alpha=0.5,
        color="black",
        label="Training (CV)",
        s=10,
        zorder=2,
    )

    if y_test_actual is not None and y_test_pred is not None:
        test_actual_plot = y_test_actual * 100.0
        test_pred_plot = y_test_pred * 100.0
        test_color = "#0080be"

        ax.scatter(
            test_actual_plot,
            test_pred_plot,
            alpha=0.7,
            color=test_color,
            label="Test",
            s=10,
            zorder=2,
        )

        if len(test_actual_plot) > 1 and np.std(test_actual_plot) > 1e-6:
            slope, intercept = np.polyfit(test_actual_plot, test_pred_plot, 1)
            x_start = max(min_val, np.min(test_actual_plot))
            x_end = min(max_val, np.max(test_actual_plot))
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

    if y_val_actual is not None and y_val_pred is not None:
        val_actual_plot = y_val_actual * 100.0
        val_pred_plot = y_val_pred * 100.0
        ax.scatter(
            val_actual_plot,
            val_pred_plot,
            alpha=0.7,
            color="#d0024a",
            marker="^",
            label="Validation",
            s=10,
            zorder=2,
        )

    ax.set_xlabel("Measured R-Selectivity [%]", fontsize=7, fontweight="bold")
    ax.set_ylabel("Predicted R-Selectivity [%]", fontsize=7, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = []
    filtered_labels = []
    for h, lab in zip(handles, labels):
        if lab not in ("1:1 Line", "Test Fit"):
            filtered_handles.append(h)
            filtered_labels.append(lab)
    if filtered_handles:
        ax.legend(filtered_handles, filtered_labels, loc="upper left", fontsize=7, frameon=False)

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
):
    os.makedirs(output_dir, exist_ok=True)

    train_r2, train_mae, train_rmse, train_mape, train_fig_cv, _, train_pred_cv = train_metrics

    if test_metrics is not None:
        test_r2, test_mae, test_rmse, test_mape, test_fig, _, test_pred_test = test_metrics
    else:
        test_r2 = test_mae = test_rmse = test_mape = None
        test_fig = None

    if val_metrics is not None:
        val_r2, val_mae, val_rmse, val_mape, val_fig, _, val_pred_val = val_metrics
    else:
        val_r2 = val_mae = val_rmse = val_mape = None
        val_fig = None

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

    with open(
        os.path.join(output_dir, "metrics.txt"),
        "w",
        encoding="utf-8",
    ) as f_metrics:
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

        if train_predictions is not None and train_actual is not None:
            train_df = pd.DataFrame(
                {
                    "Actual": train_actual,
                    "Predicted": train_predictions,
                    "Error": train_actual - train_predictions,
                    "Absolute Error": np.abs(train_actual - train_predictions),
                    "Relative Error(%)": 100
                    * np.abs(
                        (train_actual - train_predictions)
                        / np.maximum(np.abs(train_actual), 1e-8)
                    ),
                }
            )
            csv_path = os.path.join(output_dir, "train_predictions_cv.csv")
            xlsx_path = os.path.join(output_dir, "train_predictions_cv.xlsx")
            train_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            train_df.to_excel(xlsx_path, index=False)

        if train_predictions_direct is not None and train_actual is not None:
            train_direct_df = pd.DataFrame(
                {
                    "Actual": train_actual,
                    "Predicted": train_predictions_direct,
                    "Error": train_actual - train_predictions_direct,
                    "Absolute Error": np.abs(train_actual - train_predictions_direct),
                    "Relative Error(%)": 100
                    * np.abs(
                        (train_actual - train_predictions_direct)
                        / np.maximum(np.abs(train_actual), 1e-8)
                    ),
                }
            )
            csv_path = os.path.join(output_dir, "train_predictions_direct.csv")
            xlsx_path = os.path.join(output_dir, "train_predictions_direct.xlsx")
            train_direct_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            train_direct_df.to_excel(xlsx_path, index=False)

        if test_predictions is not None and test_actual is not None:
            test_df = pd.DataFrame(
                {
                    "Actual": test_actual,
                    "Predicted": test_predictions,
                    "Error": test_actual - test_predictions,
                    "Absolute Error": np.abs(test_actual - test_predictions),
                    "Relative Error(%)": 100
                    * np.abs(
                        (test_actual - test_predictions)
                        / np.maximum(np.abs(test_actual), 1e-8)
                    ),
                }
            )
            csv_path = os.path.join(output_dir, "test_predictions.csv")
            xlsx_path = os.path.join(output_dir, "test_predictions.xlsx")
            test_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            test_df.to_excel(xlsx_path, index=False)

        if val_predictions is not None and val_actual is not None:
            val_df = pd.DataFrame(
                {
                    "Actual": val_actual,
                    "Predicted": val_predictions,
                    "Error": val_actual - val_predictions,
                    "Absolute Error": np.abs(val_actual - val_predictions),
                    "Relative Error(%)": 100
                    * np.abs(
                        (val_actual - val_predictions)
                        / np.maximum(np.abs(val_actual), 1e-8)
                    ),
                }
            )
            csv_path = os.path.join(output_dir, "validation_predictions.csv")
            xlsx_path = os.path.join(output_dir, "validation_predictions.xlsx")
            val_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            val_df.to_excel(xlsx_path, index=False)

    print(f"\nResults saved to folder: {output_dir}")
    return best_params


def main():
    code_file = __file__ if "__file__" in globals() else None

    train_data = read_data("train_data.xlsx")

    # 相关性阈值：改成常量
    corr_threshold = CORR_THRESHOLD

    print("\nRunning high-correlation feature filtering...")
    X_raw = train_data.iloc[:, 2:154]
    X_filtered, to_drop = (
        remove_highly_correlated_features(X_raw, corr_threshold)
    )
    post_corr_features = X_filtered.columns.tolist()

    # PCA 保留的累计贡献率：常量
    pca_variance = PCA_VARIANCE

    print("\nRunning preprocessing (scaling + PCA)...")
    (
        X_pca,
        Y,
        scaler,
        pca,
        train_original_indices,
    ) = preprocess_data(
        train_data,
        pca_variance=pca_variance,
        dropped_cols=to_drop,
    )

    print("\nModel input summary:")
    print(f"  Features: samples={X_pca.shape[0]}, dims={X_pca.shape[1]}")
    print(f"  Targets : samples={Y.shape[0]}")

    kernels = ["linear", "rbf", "poly", "sigmoid"]
    selected_kernel = SELECTED_KERNEL.lower()
    if selected_kernel not in kernels:
        raise ValueError(f"SELECTED_KERNEL must be one of {kernels}, got {SELECTED_KERNEL!r}")

    print(f"\nUsing SVR kernel (from config): {selected_kernel}")

    if not RUN_OPTUNA:
        print("RUN_OPTUNA=False, skip hyperparameter search, exiting.")
        return

    n_trials = OPTUNA_TRIALS
    timeout = OPTUNA_TIMEOUT

    study, best_params = optimize_with_optuna_simplified(
        X_pca,
        Y.values,
        selected_kernel=selected_kernel,
        n_trials=n_trials,
        timeout=timeout,
    )

    print("\nTraining final SVR model...")
    print(f"Using best parameters: {best_params}")
    trained_model = train_model(X_pca, Y.values, best_params)

    # 作图大小 & 字体 & 纵横比
    plot_width = PLOT_WIDTH
    plot_height = PLOT_HEIGHT
    plot_figsize = (plot_width, plot_height)

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

    train_r2_cv, train_mae_cv, train_rmse_cv, train_mape_cv, fig_cv, _, train_pred_cv = (
        evaluate_model(
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
    (
        train_r2_direct,
        train_mae_direct,
        train_rmse_direct,
        train_mape_direct,
        fig_direct,
        _,
        train_pred_direct,
    ) = evaluate_model(
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

    # 测试集
    test_file = TEST_FILE_PATH or ""
    test_metrics = None
    test_predictions = None
    test_actual = None
    test_fig = None

    if test_file:
        try:
            test_data = read_data(test_file)
            print("\nProcessing test data...")
            (
                test_X_pca,
                test_Y,
                _,
                _,
                test_original_indices,
            ) = preprocess_data(
                test_data,
                is_test=True,
                scaler=scaler,
                pca=pca,
                dropped_cols=to_drop,
                pca_variance=pca_variance,
            )

            print("\nEvaluating test set (using trained SVR)...")
            (
                test_r2,
                test_mae,
                test_rmse,
                test_mape,
                test_fig,
                _,
                test_predictions,
            ) = evaluate_model(
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

    # 验证集
    val_file = VAL_FILE_PATH or ""
    val_metrics = None
    val_predictions = None
    val_actual = None
    val_fig = None

    if val_file:
        try:
            val_data = read_data(val_file)
            print("\nProcessing validation data...")
            (
                val_X_pca,
                val_Y,
                _,
                _,
                val_original_indices,
            ) = preprocess_data(
                val_data,
                is_test=True,
                scaler=scaler,
                pca=pca,
                dropped_cols=to_drop,
                pca_variance=pca_variance,
            )

            print("\nEvaluating validation set (using trained SVR)...")
            (
                val_r2,
                val_mae,
                val_rmse,
                val_mape,
                val_fig,
                _,
                val_predictions,
            ) = evaluate_model(
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

    # 输出文件夹
    folder_name = OUTPUT_FOLDER_NAME or ""
    if not folder_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        cv_label = (
            "LOOCV"
            if cv_strategy_eval == "loocv"
            else f"{n_splits_eval}KFold"
        )
        folder_name = (
            f"{timestamp}_SVR_{selected_kernel.upper()}_R2_{train_r2_cv:.3f}_{cv_label}"
        )

    output_dir = folder_name
    os.makedirs(output_dir, exist_ok=True)

    combined_plot_fig = plot_combined_predictions(
        Y.values,
        train_pred_cv,
        test_actual,
        test_predictions,
        val_actual,
        val_predictions,
        train_r2=train_r2_cv,
        train_mae=train_mae_cv,
        test_r2=test_metrics[0] if test_metrics is not None else None,
        test_mae=test_metrics[1] if test_metrics is not None else None,
        val_r2=val_metrics[0] if val_metrics is not None else None,
        val_mae=val_metrics[1] if val_metrics is not None else None,
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
    )


    optuna_path = os.path.join(output_dir, "optuna_study.joblib")
    dump(study, optuna_path)

    dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    dump(pca, os.path.join(output_dir, "pca.joblib"))
    dump(trained_model, os.path.join(output_dir, "trained_svr_model.joblib"))
    dump(to_drop, os.path.join(output_dir, "dropped_columns.joblib"))

    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        hyperparams = {
            "model_type": "SVR",
            "SVR_selected_kernel": selected_kernel,
            "SVR_best_params": best_params,
            "cv_strategy_for_evaluation": cv_strategy_eval,
            "n_splits_for_kfold_evaluation": (
                n_splits_eval if cv_strategy_eval == "kfold" else "N/A"
            ),
            "plot_figsize": plot_figsize,
            "tick_label_fontsize": tick_label_fontsize,
            "plot_aspect_ratio": aspect_ratio,
        }
        json.dump(hyperparams, f, indent=2)

    if code_file:
        shutil.copy2(code_file, os.path.join(output_dir, "code_copy.py"))

    print("\nTraining and evaluation completed.")
    print("Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()

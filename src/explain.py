"""
explain.py - SHAP explainability for the school absence Random Forest

Generates SHAP summary plot, beeswarm plot, and dependence plots for
the top 5 features. Saves all figures to outputs/figures/.

Usage:
    python src/explain.py [--model models/random_forest.pkl]
                          [--features outputs/X_test.csv]
                          [--out-dir outputs]
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# More readable labels for feature names in plots
FEATURE_LABELS = {
    "imd_score": "IMD deprivation score",
    "imd_quintile": "Deprivation quintile",
    "ofsted_numeric": "Ofsted rating (1=Outstanding)",
    "log_pupils": "School size (log pupils)",
    "phase_numeric": "School phase",
    "is_urban": "Urban location",
}


def _friendly_name(col: str) -> str:
    """Return a human-readable label for a feature column name.

    Parameters
    ----------
    col : str
        Raw column name from the feature matrix.

    Returns
    -------
    str
        Display label for use in plot axes and titles.
    """
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    if col.startswith("region_"):
        return col.replace("region_", "Region: ").replace("_", " ")
    return col.replace("_", " ").title()


def compute_shap_values(
    model, X: pd.DataFrame, sample_size: int = 500
) -> tuple:
    """Compute SHAP values for the Random Forest model.

    Uses the TreeExplainer for speed. If X has more than sample_size rows,
    a random subsample is used to keep computation time manageable.

    Parameters
    ----------
    model : sklearn RandomForestRegressor
        Fitted model.
    X : pandas.DataFrame
        Feature matrix (test set or a sample of it).
    sample_size : int, optional
        Max rows to use for SHAP computation. Default 500.

    Returns
    -------
    shap_values : numpy.ndarray
        SHAP values array, shape (n_samples, n_features).
    X_sample : pandas.DataFrame
        The (possibly subsampled) feature matrix used.
    explainer : shap.TreeExplainer
        The fitted explainer object.
    """
    if len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[idx].copy()
    else:
        X_sample = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, X_sample, explainer


def plot_shap_summary(
    shap_values: np.ndarray, X_sample: pd.DataFrame, out_dir: Path
) -> Path:
    """Save a SHAP bar summary plot (mean absolute SHAP values).

    Parameters
    ----------
    shap_values : numpy.ndarray
        SHAP values from compute_shap_values().
    X_sample : pandas.DataFrame
        Feature matrix corresponding to shap_values.
    out_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    # Rename columns for display
    X_display = X_sample.rename(columns=_friendly_name)

    fig, ax = plt.subplots(figsize=(9, 5))
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    labels = [_friendly_name(X_sample.columns[i]) for i in order]

    ax.barh(range(len(order)), mean_abs[order][::-1], color="steelblue", height=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", labelpad=8)
    ax.set_title(
        "Feature importance — mean absolute SHAP value\n"
        "Random Forest: persistent absence rate",
        fontsize=11, pad=10,
    )
    plt.tight_layout()
    path = out_dir / "shap_summary_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_shap_beeswarm(
    shap_values: np.ndarray, X_sample: pd.DataFrame, out_dir: Path
) -> Path:
    """Save a SHAP beeswarm plot.

    Parameters
    ----------
    shap_values : numpy.ndarray
        SHAP values from compute_shap_values().
    X_sample : pandas.DataFrame
        Feature matrix corresponding to shap_values.
    out_dir : Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    X_display = X_sample.rename(columns=_friendly_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_display,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    plt.title(
        "SHAP beeswarm — direction and magnitude of feature effects\n"
        "Random Forest: persistent absence rate",
        fontsize=11, pad=10,
    )
    plt.tight_layout()
    path = out_dir / "shap_beeswarm.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    out_dir: Path,
    interaction_feature: str = "auto",
) -> Path:
    """Save a SHAP dependence plot for a single feature.

    Parameters
    ----------
    shap_values : numpy.ndarray
        SHAP values array.
    X_sample : pandas.DataFrame
        Feature matrix.
    feature : str
        Column name of the feature to plot.
    out_dir : Path
        Directory to save the figure.
    interaction_feature : str, optional
        Column name for the colour interaction. Default 'auto'.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    if feature not in X_sample.columns:
        print(f"  Skipping dependence plot: {feature} not in features.")
        return None

    feat_idx = list(X_sample.columns).index(feature)
    x_vals = X_sample[feature].values
    shap_vals = shap_values[:, feat_idx]

    # Colour by a second feature if available
    color_vals = None
    color_label = None
    if interaction_feature == "auto":
        # Use IMD score as the interaction colour if available
        if "imd_score" in X_sample.columns and feature != "imd_score":
            color_vals = X_sample["imd_score"].values
            color_label = "IMD score"
    elif interaction_feature in X_sample.columns:
        color_vals = X_sample[interaction_feature].values
        color_label = _friendly_name(interaction_feature)

    fig, ax = plt.subplots(figsize=(8, 5))

    if color_vals is not None:
        sc = ax.scatter(x_vals, shap_vals, c=color_vals, cmap="RdYlGn_r",
                        s=12, alpha=0.6, linewidths=0)
        cbar = plt.colorbar(sc, ax=ax)
        if color_label:
            cbar.set_label(color_label, fontsize=9)
    else:
        ax.scatter(x_vals, shap_vals, s=12, alpha=0.6, color="steelblue", linewidths=0)

    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel(_friendly_name(feature), labelpad=8)
    ax.set_ylabel(f"SHAP value for {_friendly_name(feature)}", labelpad=8)
    ax.set_title(
        f"SHAP dependence — {_friendly_name(feature)}\n"
        "Effect on predicted persistent absence rate",
        fontsize=11, pad=10,
    )

    plt.tight_layout()
    safe_name = feature.replace("/", "_").replace(" ", "_")
    path = out_dir / f"shap_dependence_{safe_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def main(
    model_path: str = "models/random_forest.pkl",
    features_path: str = "outputs/X_test.csv",
    out_dir: str = "outputs",
) -> None:
    """Generate all SHAP plots.

    Parameters
    ----------
    model_path : str
        Path to the saved Random Forest model.
    features_path : str
        Path to X_test.csv saved by model.py.
    out_dir : str
        Root output directory; figures saved to <out_dir>/figures/.
    """
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading features from {features_path}...")
    X_test = pd.read_csv(features_path, index_col=0)

    print("Computing SHAP values (this may take a moment)...")
    shap_values, X_sample, _ = compute_shap_values(model, X_test)

    print("\n--- Generating SHAP plots ---")
    plot_shap_summary(shap_values, X_sample, fig_dir)
    plot_shap_beeswarm(shap_values, X_sample, fig_dir)

    # Dependence plots for top 5 features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(mean_abs)[::-1][:5]
    top5_features = [X_sample.columns[i] for i in top5_idx]

    print("\n--- Generating dependence plots ---")
    for feat in top5_features:
        plot_shap_dependence(shap_values, X_sample, feat, fig_dir)

    print(f"\nAll SHAP plots saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP explainability for school absence Random Forest."
    )
    parser.add_argument("--model", default="models/random_forest.pkl")
    parser.add_argument("--features", default="outputs/X_test.csv")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()
    main(model_path=args.model, features_path=args.features, out_dir=args.out_dir)

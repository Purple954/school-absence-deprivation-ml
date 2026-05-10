"""
model.py - train and evaluate the Random Forest model

Trains a Random Forest Regressor with 5-fold cross validation and
GridSearchCV hyperparameter tuning. Compares against an OLS baseline.
Saves the fitted model and a metrics summary.

Usage:
    python src/model.py [--features data/processed/features.csv]
                        [--target data/processed/target.csv]
                        [--out-dir outputs]
                        [--model-dir models]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, KFold, cross_val_score, train_test_split,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# Hyperparameter grid for GridSearchCV
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_groups(groups_path: str) -> pd.DataFrame | None:
    """Load the groups metadata saved by features.py, if it exists.

    Parameters
    ----------
    groups_path : str
        Path to groups.csv.

    Returns
    -------
    pandas.DataFrame or None
        Columns: urn, [year_key], [region_name]. Index matches features.csv.
        Returns None if the file does not exist.
    """
    path = Path(groups_path)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def load_data(features_path: str, target_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and target vector from CSV.

    Parameters
    ----------
    features_path : str
        Path to features.csv produced by features.py.
    target_path : str
        Path to target.csv produced by features.py.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable (persistent absence rate).
    """
    X = pd.read_csv(features_path, index_col=0)
    y = pd.read_csv(target_path, index_col=0).squeeze()
    # Align indices
    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train and test sets.

    When ``groups`` is provided and contains a ``urn`` column, splits by
    school so that all rows for the same school land in the same split.
    This prevents data leakage in panel data where the same school appears
    in multiple years.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    groups : pandas.DataFrame, optional
        Metadata from groups.csv. Must share the same index as X/y.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    """
    if groups is not None and "urn" in groups.columns:
        # Split by unique school URN to avoid cross-year leakage
        unique_urns = groups["urn"].unique()
        train_urns, test_urns = train_test_split(
            unique_urns, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_urns_set = set(train_urns)
        test_urns_set = set(test_urns)
        train_mask = groups["urn"].isin(train_urns_set)
        test_mask = groups["urn"].isin(test_urns_set)
        print(f"  School-level split: {train_mask.sum():,} train rows "
              f"({len(train_urns):,} schools) / {test_mask.sum():,} test rows "
              f"({len(test_urns):,} schools).")
        return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def run_ols_baseline(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    """Fit an OLS baseline and return test-set metrics.

    Uses statsmodels OLS with a constant term added. The baseline provides
    a reference point to show whether the Random Forest adds predictive
    value beyond a simple linear model.

    Parameters
    ----------
    X_train : pandas.DataFrame
    y_train : pandas.Series
    X_test : pandas.DataFrame
    y_test : pandas.Series

    Returns
    -------
    dict
        Keys: rmse, r2, model_name.
    """
    print("Fitting OLS baseline...")
    # astype(float) converts nullable Int64 to float64 so statsmodels can handle it
    X_tr_clean = X_train.astype(float).replace([np.inf, -np.inf], np.nan)
    X_tr_clean = X_tr_clean.fillna(X_tr_clean.median())
    X_te_clean = X_test.astype(float).replace([np.inf, -np.inf], np.nan)
    X_te_clean = X_te_clean.fillna(X_tr_clean.median())

    X_tr = sm.add_constant(X_tr_clean, has_constant="add")
    X_te = sm.add_constant(X_te_clean, has_constant="add")

    # Align columns after add_constant
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    ols = sm.OLS(y_train, X_tr).fit()
    preds = ols.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    print(f"  OLS baseline  — RMSE: {rmse:.3f}  R²: {r2:.3f}")
    return {"model_name": "OLS baseline", "rmse": rmse, "r2": r2}


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    region_groups: pd.Series | None = None,
) -> RandomForestRegressor:
    """Tune a Random Forest Regressor using GridSearchCV.

    When ``region_groups`` is provided, uses spatial GroupKFold CV (one fold
    per English region) so that train/validation folds never share the same
    geography. This gives an honest out-of-sample estimate because nearby
    schools are highly correlated.

    Falls back to standard KFold when region groups are not available.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training targets.
    region_groups : pandas.Series, optional
        Region label for each training row (same index as X_train).

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Best estimator fitted on all training data.
    """
    if region_groups is not None:
        n_regions = region_groups.nunique()
        cv = GroupKFold(n_splits=min(CV_FOLDS, n_regions))
        fit_kwargs = {"groups": region_groups.values}
        cv_label = f"spatial GroupKFold ({n_regions} regions)"
    else:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fit_kwargs = {}
        cv_label = f"standard KFold ({CV_FOLDS} folds)"

    print(f"Tuning Random Forest with GridSearchCV ({cv_label})...")
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        rf, PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train, **fit_kwargs)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV RMSE: {-grid.best_score_:.3f}")
    return grid.best_estimator_


def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    """Evaluate the fitted model on train and test sets.

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted model.
    X_train, y_train : training data
    X_test, y_test : held-out test data

    Returns
    -------
    dict
        Keys: model_name, train_rmse, train_r2, test_rmse, test_r2,
        cv_rmse_mean, cv_rmse_std.
    """
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1
    )

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    metrics = {
        "model_name": "Random Forest",
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
        "train_r2": float(r2_score(y_train, train_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "test_r2": float(r2_score(y_test, test_preds)),
        "cv_rmse_mean": float(-cv_scores.mean()),
        "cv_rmse_std": float(cv_scores.std()),
        "best_params": model.get_params(),
    }

    print(f"\n  Random Forest — Test RMSE: {metrics['test_rmse']:.3f}  "
          f"Test R²: {metrics['test_r2']:.3f}")
    print(f"  CV RMSE: {metrics['cv_rmse_mean']:.3f} ± {metrics['cv_rmse_std']:.3f}")
    return metrics


def feature_importances_df(model: RandomForestRegressor, feature_names: list) -> pd.DataFrame:
    """Return feature importances as a sorted DataFrame.

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted model.
    feature_names : list of str
        Column names from the feature matrix.

    Returns
    -------
    pandas.DataFrame
        Columns: feature, importance. Sorted descending.
    """
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp


def main(
    features_path: str = "data/processed/features.csv",
    target_path: str = "data/processed/target.csv",
    groups_path: str = "data/processed/groups.csv",
    out_dir: str = "outputs",
    model_dir: str = "models",
) -> None:
    """Run the full modelling pipeline.

    Parameters
    ----------
    features_path : str
        Path to features.csv.
    target_path : str
        Path to target.csv.
    groups_path : str
        Path to groups.csv produced by features.py (school/region metadata).
    out_dir : str
        Directory for metrics and plots.
    model_dir : str
        Directory to save the fitted model.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(model_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(features_path, target_path)
    groups = load_groups(groups_path)
    if groups is not None:
        groups = groups.loc[groups.index.isin(X.index)]
        print(f"Loaded {len(X):,} rows, {X.shape[1]} features, {len(groups):,} group rows.")
    else:
        print(f"Loaded {len(X):,} schools, {X.shape[1]} features (no groups file).")

    X_train, X_test, y_train, y_test = train_test_split_data(X, y, groups)
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Spatial region groups for CV (aligned to training rows)
    region_groups_train = None
    if groups is not None and "region_name" in groups.columns:
        region_groups_train = groups.loc[X_train.index, "region_name"]
        print(f"  Spatial CV using {region_groups_train.nunique()} regions.")

    # OLS baseline
    ols_metrics = run_ols_baseline(X_train, y_train, X_test, y_test)

    # Random Forest
    rf = tune_random_forest(X_train, y_train, region_groups_train)
    rf_metrics = evaluate_model(rf, X_train, y_train, X_test, y_test)

    # Feature importances
    imp_df = feature_importances_df(rf, list(X.columns))
    print("\nTop 10 features by importance:")
    print(imp_df.head(10).to_string(index=False))
    imp_df.to_csv(out / "feature_importances.csv", index=False)

    # Save metrics
    all_metrics = {
        "random_forest": rf_metrics,
        "ols_baseline": ols_metrics,
    }
    with open(out / "model_metrics.json", "w") as fh:
        json.dump(all_metrics, fh, indent=2, default=str)
    print(f"\nMetrics saved to {out / 'model_metrics.json'}")

    # Save model and test data for explain.py
    joblib.dump(rf, mdir / "random_forest.pkl")
    X_test.to_csv(out / "X_test.csv")
    y_test.to_csv(out / "y_test.csv", header=True)
    print(f"Model saved to {mdir / 'random_forest.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the school absence Random Forest model."
    )
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--target", default="data/processed/target.csv")
    parser.add_argument("--groups", default="data/processed/groups.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    main(
        features_path=args.features,
        target_path=args.target,
        groups_path=args.groups,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
    )

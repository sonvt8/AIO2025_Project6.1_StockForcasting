"""
Model Trainer Utility
Extracted training logic from export_models.py for use in main.py
"""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from app.config import DATA_FILE, FEATURE_NAMES, MODEL_CONFIG, MODEL_PATHS
from app.utils.data_processing import load_data, prepare_features_from_dataframe


def set_seed(seed=42):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)


def rolling_elasticnet_forecast(
    X, y, window_size, alpha, l1_ratio, window_type="sliding", random_state=42
):
    """Rolling forecast for validation"""
    n_samples = len(y)
    preds = np.full(n_samples, np.nan, dtype=float)

    for i in range(window_size, n_samples):
        if window_type == "sliding":
            train_slice = slice(i - window_size, i)
        else:
            train_slice = slice(0, i)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        model.fit(X[train_slice], y[train_slice])
        preds[i] = model.predict(X[i : i + 1])[0]

    return preds


def grid_search_stage2_mse(X_all_scaled, y_all, val_mask, seed=42):
    """Stage 2 grid search"""
    window_sizes = [126, 252, 504, 756]
    window_types = ["sliding", "expanding"]
    alphas = [0.0005, 0.001, 0.005, 0.01]
    l1_ratios = [0.2, 0.5, 0.8]

    records = []
    total = len(window_sizes) * len(window_types) * len(alphas) * len(l1_ratios)
    count = 0

    print("Stage 2 grid search...")
    for w in window_sizes:
        for wt in window_types:
            for a in alphas:
                for l1 in l1_ratios:
                    count += 1
                    preds_all = rolling_elasticnet_forecast(
                        X=X_all_scaled,
                        y=y_all,
                        window_size=w,
                        alpha=a,
                        l1_ratio=l1,
                        window_type=wt,
                        random_state=seed,
                    )
                    mask = val_mask & ~np.isnan(preds_all)
                    n_val = int(mask.sum())
                    if n_val == 0:
                        mse = np.nan
                    else:
                        mse = mean_squared_error(y_all[mask], preds_all[mask])

                    records.append(
                        {
                            "window_size": w,
                            "window_type": wt,
                            "alpha": a,
                            "l1_ratio": l1,
                            "n_val": n_val,
                            "mse": mse,
                        }
                    )

                    if count % 20 == 0:
                        print(f"  Progress: {count}/{total}")

    df = pd.DataFrame(records)
    df = df.sort_values("mse", ascending=True).reset_index(drop=True)
    best_row = df.iloc[0].to_dict()
    best_config = {
        "window_size": int(best_row["window_size"]),
        "window_type": best_row["window_type"],
        "alpha": float(best_row["alpha"]),
        "l1_ratio": float(best_row["l1_ratio"]),
    }
    return df, best_config


def grid_search_stage3_mse(
    X_all_scaled, y_all, val_mask, base_window_size, base_window_type, seed=42
):
    """Stage 3 grid search"""
    alphas = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]
    l1_ratios = [0.2, 0.5, 0.8]

    w = base_window_size
    wt = base_window_type
    records = []

    print("Stage 3 grid search...")
    for a in alphas:
        for l1 in l1_ratios:
            preds_all = rolling_elasticnet_forecast(
                X=X_all_scaled,
                y=y_all,
                window_size=w,
                alpha=a,
                l1_ratio=l1,
                window_type=wt,
                random_state=seed,
            )
            mask = val_mask & ~np.isnan(preds_all)
            n_val = int(mask.sum())
            if n_val == 0:
                mse = np.nan
            else:
                mse = mean_squared_error(y_all[mask], preds_all[mask])

            records.append(
                {
                    "window_size": w,
                    "window_type": wt,
                    "alpha": a,
                    "l1_ratio": l1,
                    "n_val": n_val,
                    "mse": mse,
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values("mse", ascending=True).reset_index(drop=True)
    best_row = df.iloc[0].to_dict()
    best_config = {
        "window_size": int(best_row["window_size"]),
        "window_type": best_row["window_type"],
        "alpha": float(best_row["alpha"]),
        "l1_ratio": float(best_row["l1_ratio"]),
    }
    return df, best_config


def fit_final_elasticnet(
    X_scaled, y, window_size, alpha, l1_ratio, window_type="sliding", random_state=42
):
    """Fit final model"""
    n_samples = len(y)
    if window_type == "sliding":
        start = n_samples - window_size
        X_train = X_scaled[start:]
        y_train = y[start:]
    else:
        X_train = X_scaled
        y_train = y

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_and_export_models() -> bool:
    """
    Train and export models

    Returns:
        True if successful, False otherwise
    """
    try:
        print("=" * 60)
        print("Training models from baseline V6")
        print("=" * 60)

        # Set seed
        set_seed(MODEL_CONFIG["seed"])

        # Create artifacts directory
        artifacts_dir = MODEL_PATHS["elasticnet"].parent
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nArtifacts directory: {artifacts_dir}")

        # Load data
        print("\nLoading data...")
        data_path = DATA_FILE
        if not data_path.exists():
            # Try root directory
            data_path = Path("FPT_train.csv")
            if not data_path.exists():
                print(f"❌ Error: FPT_train.csv not found at {DATA_FILE} or project root.")
                print("   Please ensure the training data file exists.")
                return False

        df = load_data(str(data_path))
        print(f"Data shape: {df.shape}")
        print(f"Time range: {df['time'].min()} -> {df['time'].max()}")

        # Prepare features
        print("\nPreparing features...")
        X_all, y_all, feat_df = prepare_features_from_dataframe(df)

        # Create validation mask
        val_start_ts = pd.Timestamp(MODEL_CONFIG["val_start"])
        feat_df["is_val"] = feat_df["time"] >= val_start_ts
        val_mask = feat_df["is_val"].values
        train_mask = ~val_mask

        print(f"Train samples: {train_mask.sum()}, Val samples: {val_mask.sum()}")
        print(f"Total features: {len(FEATURE_NAMES)}")

        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        scaler.fit(X_all[train_mask])
        X_all_scaled = scaler.transform(X_all)

        # Grid search
        print("\n" + "=" * 60)
        print("Stage 2: Finding best window_size and window_type...")
        print("=" * 60)
        results_s2, best_s2 = grid_search_stage2_mse(
            X_all_scaled, y_all, val_mask, MODEL_CONFIG["seed"]
        )
        print(f"\nBest Stage 2 config: {best_s2}")

        print("\n" + "=" * 60)
        print("Stage 3: Tuning alpha and l1_ratio...")
        print("=" * 60)
        results_s3, best_s3 = grid_search_stage3_mse(
            X_all_scaled,
            y_all,
            val_mask,
            best_s2["window_size"],
            best_s2["window_type"],
            MODEL_CONFIG["seed"],
        )
        print(f"\nBest Stage 3 config: {best_s3}")

        # Calibration
        print("\n" + "=" * 60)
        print("Calibrating model...")
        print("=" * 60)

        preds_elasticnet = rolling_elasticnet_forecast(
            X=X_all_scaled,
            y=y_all,
            window_size=int(best_s3["window_size"]),
            alpha=float(best_s3["alpha"]),
            l1_ratio=float(best_s3["l1_ratio"]),
            window_type=best_s3["window_type"],
            random_state=MODEL_CONFIG["seed"],
        )

        mask_val_used = val_mask & ~np.isnan(preds_elasticnet)
        y_true_val = y_all[mask_val_used]
        y_pred_elasticnet = preds_elasticnet[mask_val_used]

        # Fit calibration model
        cal_elasticnet = LinearRegression()
        cal_elasticnet.fit(y_pred_elasticnet.reshape(-1, 1), y_true_val)
        y_pred_elasticnet_cal = cal_elasticnet.predict(y_pred_elasticnet.reshape(-1, 1))

        print(f"Calibration: a={cal_elasticnet.intercept_:.6e}, b={cal_elasticnet.coef_[0]:.6f}")
        print(f"MSE (raw): {mean_squared_error(y_true_val, y_pred_elasticnet):.8f}")
        print(f"MSE (cal): {mean_squared_error(y_true_val, y_pred_elasticnet_cal):.8f}")

        # Ensemble optimization
        print("\n" + "=" * 60)
        print("Optimizing ensemble weight...")
        print("=" * 60)

        val_times = feat_df["time"].values[mask_val_used]
        time_to_close = dict(zip(df["time"].values, df["close"].values, strict=False))
        price_t_list = [time_to_close.get(t, np.nan) for t in val_times]
        price_t_arr = np.array(price_t_list, dtype=float)
        price_tp1_true = price_t_arr * np.exp(y_true_val)

        price_pred_naive = price_t_arr.copy()
        price_pred_model = price_t_arr * np.exp(y_pred_elasticnet_cal)

        best_w = None
        best_price_mse = np.inf

        for w in np.linspace(0.0, 1.0, 101):
            p_blend = w * price_pred_naive + (1.0 - w) * price_pred_model
            mse_blend = mean_squared_error(price_tp1_true, p_blend)
            if mse_blend < best_price_mse:
                best_price_mse = mse_blend
                best_w = w

        print(f"Best ensemble weight: w_naive={best_w:.2f}, w_model={1.0-best_w:.2f}")
        print(f"Ensemble MSE: {best_price_mse:.6f}")

        # Fit final model
        print("\n" + "=" * 60)
        print("Fitting final model...")
        print("=" * 60)

        final_elasticnet = fit_final_elasticnet(
            X_scaled=X_all_scaled,
            y=y_all,
            window_size=int(best_s3["window_size"]),
            alpha=float(best_s3["alpha"]),
            l1_ratio=float(best_s3["l1_ratio"]),
            window_type=best_s3["window_type"],
            random_state=MODEL_CONFIG["seed"],
        )

        # Save models
        print("\n" + "=" * 60)
        print("Saving models...")
        print("=" * 60)

        joblib.dump(final_elasticnet, artifacts_dir / "elasticnet_model.pkl")
        print(f"✅ Saved: {artifacts_dir / 'elasticnet_model.pkl'}")

        joblib.dump(scaler, artifacts_dir / "scaler.pkl")
        print(f"✅ Saved: {artifacts_dir / 'scaler.pkl'}")

        joblib.dump(cal_elasticnet, artifacts_dir / "calibration_model.pkl")
        print(f"✅ Saved: {artifacts_dir / 'calibration_model.pkl'}")

        # Save config
        model_config = {
            "window_size": int(best_s3["window_size"]),
            "window_type": best_s3["window_type"],
            "alpha": float(best_s3["alpha"]),
            "l1_ratio": float(best_s3["l1_ratio"]),
            "ensemble_weight": float(best_w),
        }

        with open(artifacts_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)
        print(f"✅ Saved: {artifacts_dir / 'model_config.json'}")

        print("\n" + "=" * 60)
        print("✅ Models trained and exported successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error training models: {e}")
        import traceback

        traceback.print_exc()
        return False

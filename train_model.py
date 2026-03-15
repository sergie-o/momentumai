# train_model.py
# MomentumAI — Pallet prediction model training
#
# Run this once (or whenever you want to retrain on fresh data):
#   python train_model.py
#
# Output:  outputs/models/pallet_model.pkl

import os
import numpy  as np
import pandas as pd
import joblib

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.ensemble          import GradientBoostingRegressor
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score

from generate_data import generate_synthetic_data
from preprocess    import get_feature_matrix, MODEL_FEATURES
from config        import CONFIG


def train_pallet_model(config: dict = CONFIG) -> None:
    """
    Full training pipeline for the pallet-demand prediction model.

    Steps
    -----
    1. Generate (or load) synthetic warehouse shift data.
    2. Engineer features via preprocess.get_feature_matrix().
    3. Split → train GradientBoostingRegressor → evaluate.
    4. Print feature importances and save model + reports.
    """

    _divider = "─" * 62

    print(_divider)
    print("  MomentumAI  ·  Pallet Model Training")
    print(_divider)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    os.makedirs("data/raw",       exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/reports",exist_ok=True)

    data_path = "data/raw/warehouse_shift_data.csv"
    if os.path.exists(data_path):
        print(f"\n📂  Loading data  →  {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("\n⚙   Generating synthetic training data …")
        df = generate_synthetic_data(config)
        df.to_csv(data_path, index=False)
        print(f"    Saved {len(df):,} rows  →  {data_path}")

    # ── 2. Features & target ───────────────────────────────────────────────────
    X = get_feature_matrix(df, config)
    y = df["pallets_to_order"]

    print(f"\n📊  Dataset  :  {len(df):,} rows  ·  {len(MODEL_FEATURES)} features")
    print(f"    Target   :  {y.min()} – {y.max()} pallets  (mean {y.mean():.1f})")
    print(f"\n    Features :")
    for f in MODEL_FEATURES:
        print(f"      • {f}")

    # ── 3. Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=config["random_seed"],
    )

    # ── 4. Model ───────────────────────────────────────────────────────────────
    # GradientBoostingRegressor outperforms plain RandomForest on this tabular
    # task — it better captures the non-linear interplay between volume,
    # spoilage rate, and shift type.
    model = GradientBoostingRegressor(
        n_estimators  = 350,
        learning_rate = 0.07,
        max_depth     = 5,
        subsample     = 0.85,
        min_samples_leaf = 5,
        random_state  = config["random_seed"],
    )

    print(f"\n🏋  Training GradientBoostingRegressor …")
    model.fit(X_train, y_train)

    # ── 5. Evaluation ──────────────────────────────────────────────────────────
    preds   = model.predict(X_test)
    mae     = mean_absolute_error(y_test, preds)
    rmse    = mean_squared_error(y_test, preds) ** 0.5
    r2      = r2_score(y_test, preds)

    # Within-N-pallet accuracy
    within_1 = np.mean(np.abs(preds - y_test) <= 1) * 100
    within_3 = np.mean(np.abs(preds - y_test) <= 3) * 100
    within_5 = np.mean(np.abs(preds - y_test) <= 5) * 100

    print(f"\n✅  Hold-out test results ({len(X_test):,} rows):")
    print(f"    MAE   : {mae:.2f} pallets   (avg error per shift)")
    print(f"    RMSE  : {rmse:.2f} pallets")
    print(f"    R²    : {r2:.4f}")
    print(f"\n    Prediction accuracy:")
    print(f"    ± 1 pallet  : {within_1:5.1f}%")
    print(f"    ± 3 pallets : {within_3:5.1f}%")
    print(f"    ± 5 pallets : {within_5:5.1f}%")

    # ── 6. Feature importance ──────────────────────────────────────────────────
    fi = (
        pd.DataFrame({
            "feature":    MODEL_FEATURES,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv("outputs/reports/feature_importance.csv", index=False)

    print(f"\n🔑  Top feature importances:")
    for _, row in fi.iterrows():
        bar = "█" * max(1, int(row["importance"] * 50))
        print(f"    {row['feature']:<35} {bar:.<50} {row['importance']:.4f}")

    # ── 7. Save predictions for inspection ────────────────────────────────────
    results                      = X_test.copy()
    results["actual_pallets"]    = y_test.values
    results["predicted_pallets"] = np.ceil(preds).astype(int)
    results["error"]             = results["predicted_pallets"] - results["actual_pallets"]
    results.to_csv("data/processed/model_predictions.csv", index=False)

    # ── 8. Persist model ───────────────────────────────────────────────────────
    model_path = "outputs/models/pallet_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾  Model saved  →  {model_path}")
    print(_divider)


if __name__ == "__main__":
    train_pallet_model()

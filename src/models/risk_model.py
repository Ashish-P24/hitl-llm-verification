import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


DATA_PATH = "data/processed/verification_dataset.csv"


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # Features used for prediction
    features = [
        "question_length",
        "response_length",
        "word_count",
        "uncertainty_count"
    ]

    target = "needs_human_review"

    # Check required columns
    missing_columns = [
        column for column in features + [target]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Prepare X and y
    X = df[features]
    y = df[target]

    print("\nFeature columns:")
    print(features)

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------
    # TRAIN-TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # TEST SET PREDICTIONS
    # --------------------------------------------------

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n================ MODEL EVALUATION ================")

    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if len(y_test.unique()) == 2:
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nTest ROC-AUC: {auc:.4f}")

    # --------------------------------------------------
    # 5-FOLD CROSS-VALIDATION
    # --------------------------------------------------

    print("\n================ CROSS-VALIDATION ================")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\nCross-validation scores:")
    print(cv_scores)

    print(f"\nMean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"CV Standard Deviation: {cv_scores.std():.4f}")

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        by="importance",
        ascending=False
    )

    print("\n================ FEATURE IMPORTANCE ================")
    print(importance.to_string(index=False))

    # --------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------

    model_path = "models/risk_model.joblib"

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": features
        },
        model_path
    )

    print(f"\nModel saved successfully: {model_path}")
    
if __name__ == "__main__":
    main()
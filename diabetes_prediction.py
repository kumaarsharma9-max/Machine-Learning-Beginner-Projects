import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # The dataset uses 0 as a placeholder for missing readings in several clinical metrics.
    missing_value_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    for column in missing_value_columns:
        df[column] = df[column].replace(0, pd.NA)

    df[missing_value_columns] = df[missing_value_columns].fillna(
        df[missing_value_columns].median()
    )

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y


def main() -> None:
    X, y = load_and_prepare_data("diabetes.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print("Diabetes Prediction Model")
    print("=" * 28)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Confusion Matrix:")
    print(conf_matrix)
    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=["No Diabetes", "Diabetes"]))
    print()
    print("Feature coefficients (largest positive -> higher diabetes risk):")
    coefficients = pd.Series(model.coef_[0], index=X.columns)
    print(coefficients.sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()

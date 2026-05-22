import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from arch import LogisticModel, PolyModel, RandomForestModel, KNNModel, LGBMModel
from dataset import ChurnDataset

DATA_ROOT = Path("Data/Data_v1")
SAVE_ROOT = Path("save_model")


def load_periods(start: int, end: int) -> pd.DataFrame:
    frames = []
    for i in range(start, end + 1):
        frames.append(pd.read_csv(DATA_ROOT / f"period_{i}.csv"))
    return pd.concat(frames, ignore_index=True)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    print(f"  acc: {accuracy_score(y_test, pred):.4f}")
    print(f"  F1: {f1_score(y_test, pred, zero_division=0):.4f}")
    print("  classification_report:")
    print(classification_report(y_test, pred, digits=4))
    print("  confusion_matrix:")
    print(confusion_matrix(y_test, pred))
    return pred


def main():
    df = load_periods(1, 10)
    dataset = ChurnDataset()
    X, y = dataset.get_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Train label distribution:", pd.Series(y_train).value_counts().to_dict())
    print("Test label distribution:", pd.Series(y_test).value_counts().to_dict())

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    models = [
        ("logistic", LogisticModel()),
        ("poly", PolyModel()),
        ("random_forest", RandomForestModel()),
        ("knn", KNNModel()),
        ("lgbm", LGBMModel()),
    ]

    for name, model in models:
        print("=" * 60)
        print(name)
        model.fit(X_train, y_train)
        evaluate(model, X_test, y_test)
        model.save(str(SAVE_ROOT / f"{name}.pkl"))


if __name__ == "__main__":
    main()

# train_model.py

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_save_model():
    # 1. Load dataset
    iris = load_iris()
    X = iris.data  # features: sepal length/width, petal length/width
    y = iris.target  # labels: 0,1,2 (iris classes)

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Define model
    model = LogisticRegression(max_iter=200)

    # 4. Train model
    model.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    # 6. Save model to models/iris_model.joblib
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "iris_model.joblib"
    joblib.dump(
        {
            "model": model,
            "target_names": iris.target_names,
            "feature_names": iris.feature_names,
        },
        model_path,
    )
    print(f"Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    train_and_save_model()

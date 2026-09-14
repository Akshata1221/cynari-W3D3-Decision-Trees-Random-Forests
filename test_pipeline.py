import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from train_and_log import DecisionTreeClassifier, RandomForestClassifier


@pytest.fixture
def dataset():
    data = load_breast_cancer()
    return train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )


def test_tuned_tree_max_depth(dataset):
    X_train, X_test, y_train, y_test = dataset
    max_depth_limit = 3

    model = DecisionTreeClassifier(max_depth=max_depth_limit, random_state=42)
    model.fit(X_train, y_train)

    assert (
        model.get_depth() <= max_depth_limit
    ), f"Tree depth {model.get_depth()} exceeded maximum limit of {max_depth_limit}!"


def test_random_forest_accuracy(dataset):
    X_train, X_test, y_train, y_test = dataset

    model = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    assert (
        acc >= 0.90
    ), f"Random Forest accuracy ({acc:.2f}) dropped below acceptable threshold (0.90)!"
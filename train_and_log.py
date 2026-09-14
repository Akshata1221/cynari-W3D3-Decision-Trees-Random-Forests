import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def run_pipeline():
    # 1. Initialize MLflow Experiment
    mlflow.set_experiment("W3D3_Decision_Trees_Random_Forests")

    # 2. Load Dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # 3. Define Models (Overfitted Tree vs Pruned Tree vs Random Forest)
    models = {
        "Overfitted_Tree": DecisionTreeClassifier(random_state=42),
        "Tuned_Tree": DecisionTreeClassifier(
            max_depth=3, min_samples_split=5, criterion="gini", random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=42
        ),
    }

    # 4. Train, Evaluate, and Log
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Fit Model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log Hyperparameters & Metrics to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Generate & Save Tree Diagram for Decision Trees
            if isinstance(model, DecisionTreeClassifier):
                fig, ax = plt.subplots(figsize=(14, 8))
                plot_tree(
                    model,
                    feature_names=data.feature_names,
                    class_names=data.target_names,
                    filled=True,
                    ax=ax,
                )
                fig_path = f"{model_name}_structure.png"
                plt.savefig(fig_path, bbox_inches="tight")
                plt.close()

                # Log Image Artifact to MLflow and clean up local file
                mlflow.log_artifact(fig_path)
                os.remove(fig_path)

            # Log Model Artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(
                f"[✓] Successfully logged {model_name:<16} | Accuracy: {acc:.4f} | F1: {f1:.4f}"
            )


if __name__ == "__main__":
    run_pipeline()
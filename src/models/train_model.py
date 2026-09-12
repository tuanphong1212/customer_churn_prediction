import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("customer_churn_prediction")

FEATUER_REPO_PATH = "feature_repo/feature_repo"
DATA_PATH = "data/final/train_processed.parquet"
REGISTERED_MODEL_NAME = "customer_churn_model"


def load_entity_df():
    df = pd.read_parquet(DATA_PATH)
    entity_df = df[["CustomerID", "event_timestamp", "Churn"]].copy()

    n_before = len(entity_df)
    entity_df = entity_df.drop_duplicates(subset="CustomerID" , keep="first")
    n_after = len(entity_df)
    if n_before != n_after:
        print(f"[WARN] {n_after - n_before} CustomerID trùng lặp đã bị loại bỏ")

    print(f"Phân phối nhãn Churn:\n{entity_df['Churn'].value_counts(normalize=True)}")
    return entity_df


def get_training_data():
    store = FeatureStore(repo_path=FEATUER_REPO_PATH)
    entity_df = load_entity_df()
    feature_service = store.get_feature_service(name="customer_churn_v1")

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service
    ).to_df()

    return training_df


def evaluate(y_true , y_pred , y_proba):
    return {
        "accuracy" : accuracy_score(y_true , y_pred),
        "precision" : precision_score(y_true , y_pred),
        "recall" : recall_score(y_true , y_pred),
        "f1_score" : f1_score(y_true , y_pred),
        "roc_auc" : roc_auc_score(y_true , y_proba)
    }


def train_candidate(name , pipeline , params , X_train , X_test , y_train , y_test):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        pipeline.fit(X_train , y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[: , 1]

        metrics = evaluate(y_test , y_pred , y_proba)

        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test , y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=X_test.head(3)
        )

        print(f"[{name}] " + ", ".join(f"{k}={v:.4f}" for k , v in metrics.items()))
        print(f"[{name}] model_uri = {model_info.model_uri}")

        return model_info.model_uri, metrics["roc_auc"]


def main():
    df = get_training_data()

    feature_columns = [c for c in df.columns if c not in ("CustomerID", "event_timestamp", "Churn")]
    X = df[feature_columns]
    y = df["Churn"]

    zero_var_cols = X.columns[X.std() == 0]
    print("Cột có phương sai = 0 (giá trị hằng số, vô nghĩa với model):", list(zero_var_cols))

    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print("Top 10 feature tương quan mạnh nhất với Churn:")
    print(correlations.head(10))

    X_train , X_test , y_train , y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    candidate = {
        "logistic_regression" : (
            Pipeline([
                ("scaler" , StandardScaler()),
                ("logistic_regression" , LogisticRegression(max_iter=1000))
            ]),
            {
            "model_type" : "logistic_regression",
            "max_iter" : 1000
            }
        ),
        "random_forest" : (
            Pipeline([
                ("random_forest" , RandomForestClassifier(
                    n_estimators=200 , max_depth=20 , random_state=42 , n_jobs=-1
                ))
            ]),
            {
            "model_type": "random_forest",
            "n_estimators": 200,
            "max_depth": 20
            }
        )
    }

    results = []
    for name , (pipeline , params) in candidate.items():
        model_uri , auc = train_candidate(name , pipeline , params , X_train , X_test , y_train , y_test)
        results.append((name , model_uri , auc))

    best_name , best_model_uri , best_auc = max(results , key=lambda x : x[2])
    print(f"\nBest model: {best_name} (model_uri={best_model_uri}, roc_auc={best_auc:.4f})")

    mv = mlflow.register_model(model_uri=best_model_uri , name=REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=mv.version
    )
    print(f"Registerd {REGISTERED_MODEL_NAME} v{mv.version} as @champion")

if __name__ == "__main__":
    main()

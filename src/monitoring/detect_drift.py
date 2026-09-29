import mlflow
import pandas as pd
from src.monitoring.psi import calculate_psi_categorical , calculate_psi_continuous

REFERENCE_PATH = "data/original/customer_churn_dataset-training-master.csv"
CURRENT_PATH = "data/test/customer_churn_dataset-testing-master.csv"

NUMERIC_FEATURE = [
    "Age" , "Tenure" , "Usage Frequency" , "Support Calls",
    "Payment Delay" , "Total Spend" , "Last Interaction"
]

CATEGORICAL_FEATURE = ["Gender" , "Subscription Type" , "Contract Length"]

PSI_WARNING = 0.1
PSI_ALERT = 0.25


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("customer_churn_monitoring")

    reference_df = pd.read_csv(REFERENCE_PATH).dropna()
    current_df = pd.read_csv(CURRENT_PATH).dropna()

    with mlflow.start_run(run_name="drift_check"):
        results = {}

        for col in NUMERIC_FEATURE:
            psi = calculate_psi_continuous(reference_df[col] , current_df[col])
            results[col] = psi

        for col in CATEGORICAL_FEATURE:
            psi = calculate_psi_categorical(reference_df[col] , current_df[col])
            results[col] = psi

        for col , psi in results.items():
            metric_name = f"psi_{col.replace(' ' , '_')}"
            mlflow.log_metric(metric_name , psi)
            print(f"{col:25s} PSI = {psi:.4f}")

        max_psi = max(results.values())
        drifted = {c : p for c , p in results.items() if p >= PSI_WARNING}

        mlflow.log_metric("max_psi" , max_psi)
        mlflow.log_metric("num_feature_drifted" , len(drifted))

        if max_psi >= PSI_ALERT:
            status = "ALERT"
        elif max_psi >= PSI_WARNING:
            status = "WARNING"
        else:
            status = "OK"

        mlflow.set_tag("drift_status" , status)
        print(f"\n=> Trạng thái: {status} (max PSI = {max_psi:.4f})")

        if(status == "ALERT"):
            raise SystemExit(1)

if __name__ == "__main__":
    main()
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

rm = client.get_registered_model("customer_churn_model")
print("Registered model:", rm.name)
print("Aliases (alias -> version):", rm.aliases)

mv = client.get_model_version("customer_churn_model", "1")
print("\nVersion 1 chi tiết:")
print("  run_id:", mv.run_id)
print("  status:", mv.status)
print("  source:", mv.source)
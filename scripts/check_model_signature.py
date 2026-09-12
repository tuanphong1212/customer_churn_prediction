from mlflow.models import get_model_info

info = get_model_info("runs:/42b4e39f8eee42179c86be7638bc11e0/model")
print("Signature:", info.signature)
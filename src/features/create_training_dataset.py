# import pandas as pd
# from feast import FeatureStore
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report
# )

# INPUT_PATH = "data/processed/train_processed.parquet"

# df = pd.read_parquet(INPUT_PATH)

# entity_df = df[
#     [
#     "CustomerID",
#     "event_timestamp",
#     "Churn"
#     ]
# ].copy()

# store = FeatureStore(
#     repo_path="feature_repo/feature_repo"
# )

# feature_refs = [
#     "customer_features:Age",
#     "customer_features:Tenure",
#     "customer_features:Usage Frequency",
#     "customer_features:Support Calls",
#     "customer_features:Payment Delay",
#     "customer_features:Total Spend",
#     "customer_features:Last Interaction",
# ]

# training_df = store.get_historical_features(
#     entity_df=entity_df,
#     features=feature_refs
# ).to_df()

# feature_columns = [
#     "Age",
#     "Tenure",
#     "Usage Frequency",
#     "Support Calls",
#     "Payment Delay",
#     "Total Spend",
#     "Last Interaction"
# ]

# X = training_df[feature_columns]
# y = training_df["Churn"]

# X_train , X_test , y_train , y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# model = Pipeline([
#     ("scaler", StandardScaler()),
#     ("classifier", LogisticRegression())

# ])

# model.fit(X_train , y_train)

# y_pred = model.predict(X_test)

# y_pro = model.predict_proba(X_test)[: , 1]

# auc = roc_auc_score(y_test , y_pro)

# # print(classification_report(y_test , y_pred))

# rf_model = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=20,
#     random_state=42,
#     n_jobs=-1,
# )

# rf_model.fit(X_train , y_train)

# y_pred_rf = rf_model.predict(X_test)

# y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# y_train_pred_rf = rf_model.predict(X_train)
# y_train_proba_rf = rf_model.predict_proba(X_train)[:, 1]

# # print("========== TRAIN ==========")
# # print("Accuracy :", accuracy_score(y_train, y_train_pred_rf))
# # print("Precision:", precision_score(y_train, y_train_pred_rf))
# # print("Recall   :", recall_score(y_train, y_train_pred_rf))
# # print("F1       :", f1_score(y_train, y_train_pred_rf))
# # print("ROC-AUC  :", roc_auc_score(y_train, y_train_proba_rf))


# # # TEST
# # print("\n========== TEST ==========")
# # print("Accuracy :", accuracy_score(y_test, y_pred_rf))
# # print("Precision:", precision_score(y_test, y_pred_rf))
# # print("Recall   :", recall_score(y_test, y_pred_rf))
# # print("F1       :", f1_score(y_test, y_pred_rf))
# # print("ROC-AUC  :", roc_auc_score(y_test, y_proba_rf))

# # result = permutation_importance(
# #     rf_model,
# #     X_test,
# #     y_test,
# #     scoring='roc_auc',
# #     n_repeats=5,
# #     random_state=42,
# #     n_jobs=-1
# # )

# # permutation_importance_df = pd.DataFrame({
# #     "Feature": X_test.columns,
# #     "Importance": result.importances_mean,
# #     "Std": result.importances_std
# # })

# # feature_importance = permutation_importance_df.sort_values(
# #     by='Importance',
# #     ascending=False
# # )

# # print(permutation_importance_df)

# selected_features = [
#     "Age",
#     "Support Calls",
#     "Payment Delay",
#     "Total Spend"
# ]

# X_train_selected = X_train[selected_features]
# X_test_selected = X_test[selected_features]

# rf_selected = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=20,
#     random_state=42,
#     n_jobs=-1,
# )

# rf_selected.fit(X_train_selected , y_train)

# y_pred_selected = rf_selected.predict(X_test_selected)
# y_proba_selected = rf_selected.predict_proba(X_test_selected)[: , 1]

# print("Accuracy :",
#       accuracy_score(y_test, y_pred_selected))

# print("Precision:",
#       precision_score(y_test, y_pred_selected))

# print("Recall   :",
#       recall_score(y_test, y_pred_selected))

# print("F1       :",
#       f1_score(y_test, y_pred_selected))

# print("ROC-AUC  :",
#       roc_auc_score(y_test, y_proba_selected))


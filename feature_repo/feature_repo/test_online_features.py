from feast import FeatureStore

store = FeatureStore(repo_path=".")

features = store.get_online_features(
    features=[
        "customer_features:Age",
        "customer_features:Tenure",
        "customer_features:Usage Frequency",
        "customer_features:Support Calls",
        "customer_features:Payment Delay",
        "customer_features:Total Spend",
    ],
    entity_rows=[
        {"CustomerID": 10}
    ]
)

df_features = features.to_df()

print('=' * 60)
print("ONLINE FEATURES")
print("=" * 60)

print(df_features)

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df_features.dtypes)
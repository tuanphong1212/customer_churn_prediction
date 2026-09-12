from feast import FeatureStore , FeatureService

def main():
    store = FeatureStore(repo_path=".")

    feature_service = store.get_feature_service("customer_churn_v1")

    features = store.get_online_features(
        features=feature_service,
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

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
from pathlib import Path

# KHAI BÁO ĐƯỜNG DẪN
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True , exist_ok=True)

#FIND RAW FILES
files = sorted(RAW_DIR.glob("train_period_*.csv"))

print("=" * 60)
print("DATA PREPROCESSING PIPELINE")
print("=" * 60)

print(f"Found {len(files)} raw files")

#VALIDATE INPUT
if len(files) == 0:
    raise FileNotFoundError(f"No train_period_*.csv files found in {RAW_DIR}")

#PROCESS EACH FILE
for file in files:
    
    print('\n' + '-' * 60)
    print(f"processing: {file.name}")
    print('-' * 60)

    #READ
    df = pd.read_csv(file)

    print(f"Raw shape: {df.shape}")

    #COPY
    df_processed = df.copy()

    # KIỂM TRA MISING VALUES 
    missing_rows = df_processed.isnull().any(axis=1).sum()

    print(f"Rows containing missing values: {missing_rows}")

    if missing_rows > 0:
        df_processed = df_processed.dropna()
        print("Missing values removed.")
    else:
        print("Dataset has no missing values.")

    #KIỂM TRA DUPLICATED
    duplicate_count = df_processed.duplicated(subset=["CustomerID"]).sum()

    print(f"Duplicated CustomerID: {duplicate_count}")

    if duplicate_count > 0:
        duplicates = df_processed[df_processed.duplicated(subset=["CustomerID"] , keep=False)]

        print("\nSample duplicate rows:")
        print(duplicates.sort_values("CustomerID").head())

        before_rows = len(df_processed)

        df_processed = df_processed.drop_duplicates(subset=["CustomerID"] , keep="first")

        after_rows = len(df_processed)

        print(f"\nRows removed: {before_rows - after_rows}")

    else:
        print("\nNo duplicates CustomerID found.")

    # CHUYỂN ĐỔI KIỂU DỮ LIỆU
    df_processed["CustomerID"] = df_processed["CustomerID"].astype("int64")

    int_columns = ["Age" , "Tenure" , "Usage Frequency" , "Support Calls"]
    float_columns = ["Payment Delay" , "Total Spend" , "Last Interaction"]

    for col in int_columns:
        df_processed[col] = df_processed[col].astype(int)

    for col in float_columns:
        df_processed[col] = df_processed[col].astype(float)

    df_processed["Churn"] = df_processed["Churn"].astype(int)

    # TẠO TIMESTAMP CHO FEATURE STORE
    DEFAULT_TIMESTAMP = pd.Timestamp("2026-1-1 00:00:00")

    df_processed['event_timestamp'] = DEFAULT_TIMESTAMP
    df_processed['created_timestamp'] = DEFAULT_TIMESTAMP

    #FEATURES ENGINEERING
    # Tỷ lệ giữa tuổi và thời gian sử dụng có thể liên quan đến khả năng rời bỏ dịch vụ.
    df_processed["Age_Tenure_Ratio"] = df_processed["Tenure"] / (df_processed["Age"] + 1)

    # Trung bình mỗi lần sử dụng dịch vụ, khách hàng chi bao nhiêu tiền?
    df_processed["Spend_per_Usage"] = df_processed["Total Spend"] / (df_processed["Usage Frequency"] + 1)

    # Trung bình mỗi tháng sử dụng dịch vụ, khách hàng phải gọi hỗ trợ bao nhiêu lần?
    df_processed["Support_Calls_per_Tenure"] = df_processed["Support Calls"] / (df_processed["Tenure"] + 1)

    # Phân cụm tổng chi tiêu của khách hàng 
    df_processed["Spending_Group"] = pd.cut(df_processed["Total Spend"],
                                            bins=[0 , 600 , 1000 , np.inf],
                                            labels=["Low" , "Medium" , "High"])

    # Phân cụm khách hàng theo thòi gian sử dụng dịch vụ
    df_processed["Tenure_Group"] = pd.cut(df_processed["Tenure"],
                                        bins=[0 , 12 , 24 , np.inf],
                                        labels=["New" , "Regular" , "Loyal"])

    #ONE-HOT ENCODING   
    CATEGORY_LEVELS = {
        "Gender" : ['Female' , 'Male'],
        "Subscription Type" : ["Basic", "Standard", "Premium"],
        "Contract Length" : ["Monthly", "Quarterly", "Annual"],
    }

    for col , levels in CATEGORY_LEVELS.items():
        df_processed[col] = pd.Categorical(df_processed[col] , categories=levels)

    categorical_columns = list(CATEGORY_LEVELS.keys()) + ["Spending_Group" , "Tenure_Group"]

    df_processed = pd.get_dummies(df_processed , columns=categorical_columns , dtype=int)

    #OUTPUT PATH
    output_name = file.stem + "_processed.parquet"
    output_path = PROCESSED_DIR / output_name

    #SAVE
    df_processed.to_parquet(output_path , index=False)

    print(f"Saved: {output_path}")
    print(f"Processed shape: {df_processed.shape}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)
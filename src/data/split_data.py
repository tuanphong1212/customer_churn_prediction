import os
import numpy as np
import pandas as pd
import math

INPUT_PATH = "data/original/customer_churn_dataset-training-master.csv"
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR , exist_ok=True) # nếu raw/data chưa có thì sẽ tạo nếu có rồi thì sẽ ko báo lỗi
df = pd.read_csv(INPUT_PATH)
# Số dòng ở mỗi phần
rows_per_split = math.ceil(len(df) / 10)

for i in range(10):
    start = i * rows_per_split

    end = start + rows_per_split

    part = df.iloc[start:end]

    output_path = os.path.join(OUTPUT_DIR , f"train_period_{i}.csv") # ghép data\raw với train_period_i.csv thành data\raw\train_period_i.csv

    part.to_csv(output_path , index=False) 

    print(f"Saved: {output_path}")



import sqlite3

TABLE_NAME = "customer_churn_prediction_customer_features"

conn = sqlite3.connect("data/online_store.db")
cursor = conn.cursor()

cursor.execute(f"""SELECT COUNT(*) FROM {TABLE_NAME}""")

count = cursor.fetchone()[0]

print("=" * 60)
print("NUMBER OF RECORDS")
print("=" * 60)
print(count)

cursor.execute(f"""
SELECT DISTINCT feature_name
FROM {TABLE_NAME};
""")

features = cursor.fetchall()

print("\n" + "=" * 60)
print("FEATURES")
print("=" * 60)

for feature in features:
    print(feature[0])

cursor.execute(f"""
SELECT *
FROM {TABLE_NAME}
LIMIT 10;
""")

rows = cursor.fetchall()

print("\n" + "=" * 60)
print("FIRST 10 RECORDS")
print("=" * 60)

for row in rows:
    print(row)


conn.close()
from datetime import timedelta

from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    Project,
)

from feast.types import (
    Float64,
    Int64,
)

project = Project(
    name="customer_churn_prediction",
    description="Feature Store for Customer Churn Prediction"
)

customer = Entity(
    name="customer",
    join_keys=["CustomerID"]
)

customer_source = FileSource(
    name="Customer_source",
    path="../../data/processed/train_processed.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

customer_feature_view = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=365),
    source=customer_source,
    online=True,
    schema = [
        #Features có sẵn
        Field(name="Age" , dtype=Int64),
        Field(name="Tenure" , dtype=Int64),
        Field(name="Usage Frequency" , dtype=Int64),
        Field(name="Support Calls" , dtype=Int64),
        Field(name="Payment Delay" , dtype=Int64),
        Field(name="Total Spend" , dtype=Float64),
        Field(name="Last Interaction" , dtype=Float64),
        #Features Engineering
        Field(name="Age_Tenure_Ratio" , dtype=Float64),
        Field(name="Spend_per_Usage" , dtype=Float64),
        Field(name="Support_Calls_per_Tenure" , dtype=Float64),
        #Features one_hot endcoding
        Field(name="Gender_Female" , dtype=Int64),
        Field(name="Gender_Male" , dtype=Int64),
        Field(name="Subscription Type_Basic" , dtype=Int64),
        Field(name="Subscription Type_Premium" , dtype=Int64),
        Field(name="Subscription Type_Standard" , dtype=Int64),
        Field(name="Contract Length_Annual" , dtype=Int64),
        Field(name="Contract Length_Monthly" , dtype=Int64),
        Field(name="Contract Length_Quarterly" , dtype=Int64),
        Field(name="Spending_Group_Low" , dtype=Int64),
        Field(name="Spending_Group_Medium" , dtype=Int64),
        Field(name="Spending_Group_High" , dtype=Int64),
        Field(name="Tenure_Group_New" , dtype=Int64),
        Field(name="Tenure_Group_Regular" , dtype=Int64),
        Field(name="Tenure_Group_Loyal" , dtype=Int64),
    ]
)



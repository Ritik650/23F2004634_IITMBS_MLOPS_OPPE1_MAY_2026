from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# Deliverable 3: entity is the stock
stock = Entity(name="stock_name", value_type=ValueType.STRING, description="stock ticker")

src = FileSource(
    name="stock_source",
    path="data/stock_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# feature views for rolling_avg_10 and volume_sum_10 (+ target for training)
stock_fv = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=src,
    online=True,
)

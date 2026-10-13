"""Example demonstrating Memory-Efficient Streaming Iterators (iter_all) in WMilvus."""

from typing import List
from pydantic import BaseModel, Field
from wmilvus import FieldVector, MetricType, WMilvus

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "db_name": "default",
}


class SensorReading(BaseModel):
    """Pydantic model representing telemetry vector embeddings."""

    __tablename__ = "SensorTelemetry"

    id: str = Field(..., description="Sensor unique observation ID")
    sensor_id: str = Field(..., description="Hardware sensor identifier")
    status: str = Field(..., description="Operational status: online, degraded, offline")
    temperature: float = Field(..., description="Measured temperature in Celsius")
    features: List[float] = FieldVector(dim=64, metric_type=MetricType.L2)


def main() -> None:
    print("--- Memory-Efficient Streaming (iter_all) in WMilvus ---")

    with WMilvus(SensorReading, milvus_config) as db:
        # Ingest sample batch of sensor data
        readings = [
            SensorReading(
                id=f"sensor_obs_{i:04d}",
                sensor_id=f"device_{i % 5}",
                status="online" if i % 2 == 0 else "degraded",
                temperature=20.0 + (i * 0.5),
                features=[float(i) * 0.01] * 64,
            )
            for i in range(20)
        ]
        db.insert_batch(readings, batch_size=10)
        print("Ingested 20 sensor telemetry observations.")

        # 1. Stream all records in chunks of 5 using iter_all
        total_streamed = 0
        print("\nStreaming records with batch_size=5:")
        for batch_index, batch in enumerate(db.iter_all(batch_size=5)):
            total_streamed += len(batch)
            print(f" Chunk {batch_index + 1}: yielded {len(batch)} records (e.g. {batch[0].id})")

        print(f"Total streamed across all chunks: {total_streamed}")

        # 2. Stream with a server-side filter expression
        print("\nStreaming online sensors with filter_expr:")
        for batch in db.iter_all(batch_size=5, filter_expr='status == "online"'):
            for item in batch:
                print(f" - {item.id}: status={item.status}, temp={item.temperature:.1f}C")


if __name__ == "__main__":
    main()

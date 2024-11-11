from kafka import KafkaProducer
import pandas as pd
import json
import time

# Load dataset
df = pd.read_csv("fraud_detection_project/dataset/fraudTest.csv")

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Send transactions to Kafka
for _, row in df.iterrows():
    transaction = row.to_dict()
    producer.send('credit_card_transactions', value=transaction)
    print("Sent transaction:", transaction)
    time.sleep(2)

producer.send('credit_card_transactions', value={"end": True})
print("\nSent end-of-stream message\n")
producer.flush()


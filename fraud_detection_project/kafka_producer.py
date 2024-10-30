
from kafka import KafkaProducer
import pandas as pd
import json
import time

# Load sample data to simulate real-time transactions
df = pd.read_csv("fraud_detection_project/creditcard.csv")

# Kafka Producer
producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda x: json.dumps(x).encode('utf-8'))

# Simulate real-time transaction streaming
for _, row in df.iterrows():
    transaction = row.to_dict()
    producer.send('credit_card_transactions', value=transaction)
    print("\n Sent transaction: \n", transaction)
    time.sleep(2)  # Simulating a 2-second delay between transactions

producer.flush()

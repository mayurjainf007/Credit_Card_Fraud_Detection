from kafka import KafkaProducer
import pandas as pd
import json
import time
import os

def status():
    return True

def main(file_name=None):
    # Define the path to the data folder and input file
    data_folder = "fraud_detection/data"
    file_path = os.path.join(data_folder, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File '{file_name}' not found.")
        return

    # Load the data from the CSV file
    df = pd.read_csv(file_path)

    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

    # Stream transactions to Kafka
    for _, row in df.iterrows():
        producer.send('credit_card_transactions', value=row.to_dict())
        time.sleep(2)  # Simulate real-time transaction flow
        print(f"Transaction sent: {row.to_dict()}")

    # Flush the producer to ensure all messages are sent
    producer.flush()

if __name__ == "__main__":
    import sys
    file_name = sys.argv[1] if len(sys.argv) > 1 else None
    main(file_name)

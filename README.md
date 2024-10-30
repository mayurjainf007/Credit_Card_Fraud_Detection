# Real-Time Credit Card Fraud Detection

This project demonstrates how to perform real-time credit card fraud detection using Apache Kafka, Spark Streaming, and Flask. The system dynamically adapts to schema changes in the input dataset, detects fraudulent transactions in real-time, and displays the results on a web dashboard.

## Project Overview
1. **Kafka Producer**: Simulates real-time transaction data by streaming credit card transactions to Kafka.
2. **Spark Consumer**: Reads streaming data from Kafka, applies the trained fraud detection model, and filters fraudulent transactions.
3. **Flask Dashboard**: Displays detected fraud transactions in real-time and allows users to download them as a CSV file.

## Prerequisites
- **Python**: 3.x
- **Apache Kafka**: Download from [Kafka's official website](https://kafka.apache.org/downloads).
- **Apache Spark**: Ensure you have Spark installed with Structured Streaming capabilities.
- **Python Packages**:
  - **Flask**: `pip install flask`
  - **PySpark**: `pip install pyspark`
  - **Kafka-Python**: `pip install kafka-python`
- **Dataset**: A credit card transactions dataset (e.g., `creditcard.csv`) should be placed in the `fraud_detection_project` directory.

## Folder Structure

```
fraud_detection_project/
├── creditcard.csv                 # Dataset file for training and streaming
├── README.md                      # Instructions and project overview
├── kafka_producer.py              # Streams transactions to Kafka
├── model.py                       # Model training script
├── spark_consumer.py              # Streaming consumer for real-time detection
├── dashboard.py                   # Flask server for the dashboard
└── templates/
    └── dashboard.html             # HTML template for the dashboard UI
```

## Setup and Execution

### 1. Start Kafka
   - **Start Zookeeper** (needed for Kafka):
     ```bash
     bin/zookeeper-server-start.sh config/zookeeper.properties
     ```
   - **Start Kafka Server**:
     ```bash
     bin/kafka-server-start.sh config/server.properties
     ```

### 2. Create Kafka Topic
Create a topic called `credit_card_transactions` for the transaction stream.
   ```bash
   bin/kafka-topics.sh --create --topic credit_card_transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
   ```

### 3. Train and Save the Models
Run the following command to train the StandardScaler and Logistic Regression models. This script will also dynamically infer the schema based on `creditcard.csv`.

   ```bash
   spark-submit fraud_detection_project/model.py
   ```

   - **What it Does**:
     - **Schema Inference**: Reads and infers the schema from `creditcard.csv`.
     - **Feature Scaling**: Trains a scaler model on the features and saves it as `scaler_model`.
     - **Logistic Regression**: Trains the logistic regression model and saves it as `fraud_detection_model`.

   - **Output**: Saved models (`scaler_model` and `fraud_detection_model`) in the `fraud_detection_project` directory.

### 4. Run Kafka Producer
Start the Kafka producer script to simulate real-time transaction streaming from `creditcard.csv`.

   ```bash
   python fraud_detection_project/kafka_producer.py
   ```

   - **What it Does**:
     - Reads each transaction row from `creditcard.csv` and sends it to the Kafka topic `credit_card_transactions`.
     - Simulates real-time streaming with a delay between each transaction.

### 5. Run Spark Streaming Consumer with Dynamic Schema
Start the Spark consumer to read from Kafka, scale features, detect fraud, and send results to the Flask dashboard.

   ```bash
   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 fraud_detection_project/spark_consumer.py
   ```

   - **Explanation of Key Functions**:
     - **Dynamic Schema**: Reads and adapts to the schema from the data at runtime.
     - **Real-Time Processing**: Reads Kafka data in micro-batches, applies scaling and model inference, and filters fraudulent transactions.
     - **Flask API Posting**: Sends each detected fraud transaction to the Flask dashboard API.

   - **Expected Output**: Fraudulent transactions will be sent to the Flask dashboard for real-time display.

### 6. Run the Flask Dashboard
Start the Flask server to view real-time fraud detection results.

   ```bash
   python fraud_detection_project/dashboard.py
   ```

   - **What it Does**:
     - Launches a web server on `http://localhost:5000`.
     - Displays fraud transactions as they are detected and allows CSV download of flagged transactions.

   - **Access URL**: Open a web browser and go to [http://localhost:5000](http://localhost:5000).

### 7. Flask Dashboard Usage
   - **Real-Time Fraud Detection Table**: Shows detected fraud transactions in a table format.
   - **CSV Download**: Click the “Download Fraudulent Transactions CSV” button to download detected fraud transactions as a CSV file.

## Example Commands Summary
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create Kafka topic
bin/kafka-topics.sh --create --topic credit_card_transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Train and save models
spark-submit fraud_detection_project/model.py

# Start Kafka producer
python fraud_detection_project/kafka_producer.py

# Start Spark consumer
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 fraud_detection_project/spark_consumer.py

# Start Flask dashboard
python fraud_detection_project/dashboard.py
```

## Notes
- Adjust file paths and Kafka configurations as needed for your setup.
- Ensure that the dataset `creditcard.csv` is correctly placed in the `fraud_detection_project` directory.

With these steps, your real-time fraud detection pipeline will be fully operational, displaying results on a dynamic dashboard and offering CSV downloads for detected fraudulent transactions.

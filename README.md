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
root/
├── project/
│   ├── data/
│   │   ├── test_data.csv
│   │   └── creditcard.csv
│   ├── models/
│   │   ├── pyspark_logistic_regression
│   │   ├── pyspark_scaler
│   │   └── fraud_detection_model.pkl
│   ├── output/
│   │   └── fraudulent_transaction.csv
│   ├── static/
│   │   ├── roc_curve.png
│   │   ├── amount_distribution.png
│   │   ├── feature_importance.png
│   │   ├── anomaly_detection.png
│   │   ├── feature_importance.csv
│   │   └── transactions_with_anomalies.csv
│   ├── templates/
│   │   └── dashboard.html
│   ├── dashboard.py
│   ├── producer.py
│   ├── consumer.py
│   ├── model.py
│   ├── result.py
├── ResearchPaper.pdf
├── enhancement.txt
├── Readme.md

## Setup and Execution

### 1. Start Kafka
   - **Start Zookeeper** (needed for Kafka):
     ```bash
     $KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties
     ```
   - **Start Kafka Server**:
     ```bash
     $KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
     ```

### 2. Create Kafka Topic
Create a topic called `credit_card_transactions` for the transaction stream.
   ```bash
   $KAFKA_HOME/bin/kafka-topics.sh --create --topic credit_card_transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
   ```

### 3. Train and Save the Models
Run the following command to train the StandardScaler and Logistic Regression models. This script will also dynamically infer the schema based on `creditcard.csv`.

   ```bash
   spark-submit fraud_detection/model.py
   ```

   - **What it Does**:
     - **Schema Inference**: Reads and infers the schema from `creditcard.csv`.
     - **Feature Scaling**: Trains a scaler model on the features and saves it as `scaler_model`.
     - **Logistic Regression**: Trains the logistic regression model and saves it as `fraud_detection_model`.

   - **Output**: Saved models (`scaler_model` and `fraud_detection_model`) in the `fraud_detection_project` directory.

### 4. Run Kafka Producer
Start the Kafka producer script to simulate real-time transaction streaming from `creditcard.csv`.

   ```bash
   python fraud_detection/producer.py
   ```

   - **What it Does**:
     - Reads each transaction row from `creditcard.csv` and sends it to the Kafka topic `credit_card_transactions`.
     - Simulates real-time streaming with a delay between each transaction.

### 5. Run Spark Streaming Consumer with Dynamic Schema
Start the Spark consumer to read from Kafka, scale features, detect fraud, and send results to the Flask dashboard.

   ```bash
   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 fraud_detection/consumer.py
   ```

   - **Explanation of Key Functions**:
     - **Dynamic Schema**: Reads and adapts to the schema from the data at runtime.
     - **Real-Time Processing**: Reads Kafka data in micro-batches, applies scaling and model inference, and filters fraudulent transactions.
     - **Flask API Posting**: Sends each detected fraud transaction to the Flask dashboard API.

   - **Expected Output**: Fraudulent transactions will be sent to the Flask dashboard for real-time display.

### 6. Run the Flask Dashboard
Start the Flask server to view real-time fraud detection results.

   ```bash
   python fraud_detection/dashboard.py
   ```

   - **What it Does**:
     - Launches a web server on `http://localhost:5000`.
     - Displays fraud transactions as they are detected and allows CSV download of flagged transactions.

   - **Access URL**: Open a web browser and go to [http://localhost:5000](http://localhost:5000).

### 7. Flask Dashboard Usage
   - **Real-Time Fraud Detection Table**: Shows detected fraud transactions in a table format.
   - **CSV Download**: Click the “Download Fraudulent Transactions CSV” button to download detected fraud transactions as a CSV file.

```

## Notes
- Adjust file paths and Kafka configurations as needed for your setup.
- Ensure that the dataset `creditcard.csv` is correctly placed in the `fraud_detection_project` directory.

With these steps, your real-time fraud detection pipeline will be fully operational, displaying results on a dynamic dashboard and offering CSV downloads for detected fraudulent transactions.

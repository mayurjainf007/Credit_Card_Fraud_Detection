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
'''
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
'''
---

## **Setup and Execution Flow**

### **1. Environment Setup**
1. **Activate Virtual Environment**:
   Ensure all dependencies are isolated within a virtual environment for consistency.
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   Verify all necessary Python and Spark libraries are installed. Use `requirements.txt` if available.
   ```bash
   pip install -r requirements.txt
   ```

---

### **2. Model Training**
1. **Train and Save Models**:
   Run the following command to:
   - Train the PySpark `StandardScaler` model for feature scaling.
   - Train the Scikit-learn `Logistic Regression` model for fraud detection.
   - Generate performance metrics (e.g., AUC-ROC curve, feature importance).
   ```bash
   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 fraud_detection/model.py
   ```

   **Outputs**:
   - `scaler_model` and `fraud_detection_model` saved to the `fraud_detection_project` directory.
   - Visualization files (AUC-ROC curve, feature importance) generated for evaluation.

---

### **3. Start Kafka Services**
1. **Start Zookeeper**:
   ```bash
   $KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties
   ```

2. **Start Kafka Server**:
   ```bash
   $KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
   ```

3. **Create Kafka Topic**:
   Define a topic for streaming transactions (`credit_card_transactions`):
   ```bash
   $KAFKA_HOME/bin/kafka-topics.sh --create --topic credit_card_transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
   ```

---

### **4. Data Streaming and Processing**
1. **Start Flask Dashboard**:
   Launch the web server to monitor real-time fraud detection.
   ```bash
   python fraud_detection/dashboard.py
   ```
   **URL**: [http://localhost:5000](http://localhost:5000)

2. **Upload Data**:
   Use the Flask dashboard to upload the dataset (`creditcard.csv`). This action triggers:
   - The **Kafka Producer** to simulate transaction streaming.
   - The **Spark Consumer** to process transactions and predict fraud.

   **Producer**:
   Sends data row-by-row to the `credit_card_transactions` Kafka topic.

   **Consumer**:
   Reads from the Kafka topic, applies scaling and predictions, and flags fraudulent transactions in real-time.

---

### **5. Real-Time Monitoring**
1. **Fraud Detection Table**:
   View flagged fraudulent transactions on the Flask dashboard.

2. **Download Results**:
   Download a CSV file of flagged transactions using the "Download Fraudulent Transactions CSV" button.

---

### **Notes**
- **Dynamic Schema Handling**: The Spark consumer dynamically infers the schema from the data.
- **Scalability**: Adjust Kafka topic configurations for higher throughput or additional partitions.
- **Error Handling**: Ensure proper exception handling in all scripts for robust performance.
- **Testing**: Test end-to-end functionality with a small subset of `creditcard.csv` before full-scale execution.

---

## **Summary**
By following these steps, your real-time fraud detection pipeline will be operational, seamlessly integrating Spark, Kafka, and Flask components for both processing and visualization.

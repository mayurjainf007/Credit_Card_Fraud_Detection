from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, DoubleType
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
import threading
import os
import pandas as pd

if not os.path.exists("fraud_detection/output"):
    os.mkdir("fraud_detection/output")

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionConsumer").getOrCreate()

# Define schema for Kafka messages
schema = StructType() \
    .add("Time", DoubleType()) \
    .add("V1", DoubleType()) \
    .add("V2", DoubleType()) \
    .add("V3", DoubleType()) \
    .add("V4", DoubleType()) \
    .add("V5", DoubleType()) \
    .add("V6", DoubleType()) \
    .add("V7", DoubleType()) \
    .add("V8", DoubleType()) \
    .add("V9", DoubleType()) \
    .add("V10", DoubleType()) \
    .add("V11", DoubleType()) \
    .add("V12", DoubleType()) \
    .add("V13", DoubleType()) \
    .add("V14", DoubleType()) \
    .add("V15", DoubleType()) \
    .add("V16", DoubleType()) \
    .add("V17", DoubleType()) \
    .add("V18", DoubleType()) \
    .add("V19", DoubleType()) \
    .add("V20", DoubleType()) \
    .add("V21", DoubleType()) \
    .add("V22", DoubleType()) \
    .add("V23", DoubleType()) \
    .add("V24", DoubleType()) \
    .add("V25", DoubleType()) \
    .add("V26", DoubleType()) \
    .add("V27", DoubleType()) \
    .add("V28", DoubleType()) \
    .add("Amount", DoubleType()) \
    .add("Class", DoubleType())

# Load PySpark ML model and scaler
lr_model = LogisticRegressionModel.load("fraud_detection/models/pyspark_logistic_regression")
scaler_model = StandardScalerModel.load("fraud_detection/models/pyspark_scaler")

# Define the CSV file path
output_csv_file = "fraud_detection/output/fraudulent_transactions.csv"

# Initialize the CSV file if it doesn't exist
if not os.path.exists(output_csv_file):
    pd.DataFrame(columns=["Time", "Amount", "Prediction"]).to_csv(output_csv_file, index=False)

# Read Kafka stream
transactions = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "credit_card_transactions") \
    .option("failOnDataLoss", "false") \
    .load()

# Deserialize Kafka messages
transactions = transactions.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Preprocess the data for fraud detection
assembler = VectorAssembler(inputCols=[col for col in schema.fieldNames() if col != "Class"], outputCol="features")
transactions = assembler.transform(transactions)
transactions = scaler_model.transform(transactions)

# Perform fraud detection
predictions = lr_model.transform(transactions)
fraudulent_transactions = predictions.filter(col("prediction") == 1)

# Start writing fraudulent transactions to the CSV file
query = fraudulent_transactions.select("Time", "Amount", "prediction") \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()


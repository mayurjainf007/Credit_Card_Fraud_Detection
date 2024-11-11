
import threading
import time
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, DoubleType, StringType
import requests
import json

# Initialize Spark session
spark = SparkSession.builder.appName("RealTimeFraudDetection").getOrCreate()

# Define schema based on dataset
schema = StructType() \
    .add("trans_date_trans_time", StringType()) \
    .add("cc_num", StringType()) \
    .add("merchant", StringType()) \
    .add("category", StringType()) \
    .add("amt", DoubleType()) \
    .add("first", StringType()) \
    .add("last", StringType()) \
    .add("is_fraud", DoubleType())

# Load models from new model directory
logistic_model = LogisticRegressionModel.load("fraud_detection_project/model/logisticregression")
rf_model = RandomForestClassificationModel.load("fraud_detection_project/model/randomforest")
scaler_model = StandardScalerModel.load("fraud_detection_project/model/scalermodel")

# Read data from Kafka
transactions = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "credit_card_transactions") \
    .load()

# Deserialize Kafka data
transactions = transactions.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Cast columns to DoubleType and handle nulls
feature_cols = [col for col in transactions.columns if col != "is_fraud"]
for col_name in feature_cols:
    transactions = transactions.withColumn(col_name, col(col_name).cast(DoubleType()))
transactions = transactions.na.fill(0, subset=feature_cols)

# Assemble and scale features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
transaction_features = assembler.transform(transactions)
scaled_data = scaler_model.transform(transaction_features)

# Predictions using both models
predictions_lr = logistic_model.transform(scaled_data)
predictions_rf = rf_model.transform(scaled_data)

# Post fraudulent transactions detected by both models
def post_fraud_transactions(batch_df, batch_id):
    try:
        fraud_data_lr = batch_df.filter(batch_df["prediction"] == 1)  # Logistic Regression model
        fraud_data_rf = batch_df.filter(batch_df["prediction"] == 1)  # Random Forest model
        fraud_data = fraud_data_lr.union(fraud_data_rf).distinct()
        
        for row in fraud_data.collect():
            data = {
                "trans_date_trans_time": row["trans_date_trans_time"],
                "amt": row["amt"],
                "prediction": row["prediction"],
                "first": row["first"],
                "last": row["last"],
                "category": row["category"],
                "merchant": row["merchant"]
            }
            try:
                response = requests.post("http://localhost:5000/add_transaction", json=data)
                if response.status_code == 200:
                    print("Transaction sent:", data)
                else:
                    print("Failed to send transaction:", response.text)
            except Exception as e:
                print("Error posting transaction:", e)
    except Exception as e:
        print(f"Error processing batch {batch_id}: {e}")

# Start stream processing with batch processing function
fraud_transactions = predictions_lr.filter(col("prediction") == 1)
query = fraud_transactions.writeStream.foreachBatch(post_fraud_transactions).outputMode("append").start()
query.awaitTermination()

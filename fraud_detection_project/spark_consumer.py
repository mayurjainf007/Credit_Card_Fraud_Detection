from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType
import requests
import json

# Step 1: Initialize Spark session
spark = SparkSession.builder \
    .appName("RealTimeFraudDetection") \
    .getOrCreate()

# Step 2: Load the schema dynamically from sample data or configuration
sample_data = spark.read.csv("fraud_detection_project/creditcard.csv", header=True, inferSchema=True)
schema = StructType.fromJson(json.loads(sample_data.schema.json()))

# Step 3: Load the trained ML model and scaler model
model = LogisticRegressionModel.load("fraud_detection_project/fraud_detection_model")
scaler_model = StandardScalerModel.load("fraud_detection_project/scaler_model")

# Step 4: Read real-time transactions from Kafka
transactions = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "credit_card_transactions") \
    .load()

# Step 5: Deserialize Kafka value and apply schema
transactions = transactions.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Step 6: Dynamically identify feature columns (excluding label column if present)
label_col = "Class" if "Class" in transactions.columns else None
feature_cols = [col for col in transactions.columns if col != label_col]

# Step 7: Feature engineering with VectorAssembler
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
transaction_features = assembler.transform(transactions)

# Step 8: Apply the pre-fitted scaler model
scaled_data = scaler_model.transform(transaction_features)

# Step 9: Apply the trained model to make predictions
predictions = model.transform(scaled_data)

# Step 10: Filter fraudulent transactions and post them to the Flask server
def post_fraud_transactions(batch_df, batch_id):
    for row in batch_df.collect():
        data = {"Time": row["Time"], "Amount": row["Amount"], "Prediction": row["prediction"]}
        try:
            response = requests.post("http://localhost:5000/add_transaction", json=data)
            if response.status_code == 200:
                print("Fraudulent transaction sent:", data)
            else:
                print("Failed to send transaction:", response.text)
        except Exception as e:
            print("Error posting transaction:", e)

fraud_transactions = predictions.filter(col("prediction") == 1)

# Step 11: Write stream to console and trigger Flask update
query = fraud_transactions \
    .writeStream \
    .foreachBatch(post_fraud_transactions) \
    .outputMode("append") \
    .start()

# Step 12: Await termination
query.awaitTermination()


import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DoubleType
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
import os

def main(file_name=None):
    # Define the input and output paths
    input_file_path = f"fraud_detection/data/{file_name}"
    output_file_path = "fraud_detection/output/fraudulent_transactions.csv"

    # Check if the input file exists
    if not os.path.exists(input_file_path):
        print(f"File '{file_name}' not found.")
        return

    # Initialize Spark session
    spark = SparkSession.builder.appName("FraudDetectionStandalone").getOrCreate()

    # Define the schema for the input data
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

    # Read the input CSV file into a Spark DataFrame
    transactions = spark.read.csv(input_file_path, header=True, schema=schema)

    # Preprocess the data for fraud detection
    assembler = VectorAssembler(inputCols=[col for col in schema.fieldNames() if col != "Class"], outputCol="features")
    transactions = assembler.transform(transactions)
    transactions = scaler_model.transform(transactions)

    # Perform fraud detection
    predictions = lr_model.transform(transactions)
    fraudulent_transactions = predictions.filter(predictions["prediction"] == 1)

    # Convert fraudulent transactions to Pandas and save to a CSV file
    fraudulent_transactions = fraudulent_transactions.select("Time", "Amount", "prediction")
    fraudulent_pandas_df = fraudulent_transactions.toPandas()
    fraudulent_pandas_df.columns = ["Time", "Amount", "Prediction"]

    # Save to the CSV file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    fraudulent_pandas_df.to_csv(output_file_path, index=False)
    print(f"Fraudulent transactions saved to '{output_file_path}'.")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    import sys
    file_name = sys.argv[1] if len(sys.argv) > 1 else None
    if file_name:
        main(file_name)
    else:
        print("Please provide the filename as an argument.")


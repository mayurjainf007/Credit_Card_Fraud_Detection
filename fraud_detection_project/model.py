from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FraudDetectionModelTraining") \
    .getOrCreate()

# Load the dataset and infer schema
data = spark.read.csv("fraud_detection_project/creditcard.csv", header=True, inferSchema=True)

# Dynamically identify feature columns (excluding the label column if present)
label_col = "Class" if "Class" in data.columns else None
feature_cols = [col for col in data.columns if col != label_col]

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

# Save the scaler model for use in the streaming application
scaler_model.write().overwrite().save("fraud_detection_project/scaler_model")

# Split the data into train and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train a Logistic Regression model
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol=label_col)
model = lr.fit(train_data)

# Evaluate the model
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)

# Save the trained model
model.write().overwrite().save("fraud_detection_project/fraud_detection_model")

print(f"ROC-AUC Score: {roc_auc:.4f}")

# Stop the Spark session
spark.stop()


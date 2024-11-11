from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.sql.functions import col
import sys

# Initialize Spark session
spark = SparkSession.builder.appName("FraudPrediction").getOrCreate()

# Load test dataset directly into a Spark DataFrame
spark_df = spark.read.csv("fraud_detection_project/dataset/fraudTest.csv", header=True, inferSchema=True)

# Load models using Spark ML's model loading
try:
    logistic_model = LogisticRegressionModel.load("fraud_detection_project/model/logisticregression")
    rf_model = RandomForestClassificationModel.load("fraud_detection_project/model/randomforest")
    scaler_model = StandardScalerModel.load("fraud_detection_project/model/scalermodel")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Define feature columns by excluding unsupported types and non-feature columns
feature_cols = [col for col in spark_df.columns if col not in ["id", "is_fraud", "first", "last", "street", "city", "state", "job", "dob", "trans_num", 
                                                               "trans_date_trans_time", "merchant", "category", "gender"]]

# Preprocess data: Fill NA values with 0 for numerical columns only
spark_df = spark_df.fillna(0, subset=feature_cols)

# Assemble features
try:
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    spark_df = assembler.transform(spark_df)
    print("Schema after VectorAssembler:")
    spark_df.printSchema()

    # Display a sample of the features column to verify correct format
    print("Sample features vector after VectorAssembler:")
    spark_df.select("features").show(5, truncate=False)
except Exception as e:
    print(f"Error during VectorAssembler transformation: {e}")
    sys.exit(1)

# Apply StandardScalerModel transformation
try:
    spark_df = scaler_model.transform(spark_df)
    print("Schema after StandardScaler transformation:")
    spark_df.printSchema()

    # Check for null values in the features column after scaling
    if spark_df.filter(spark_df["features"].isNull()).count() > 0:
        print("Error: Null values detected in 'features' column after scaling.")
        sys.exit(1)
except Exception as e:
    print(f"Error during StandardScaler transformation: {e}")
    sys.exit(1)

# Use a small sample of data for predictions to simplify debugging
spark_df_sample = spark_df.limit(10)

# Perform prediction with Logistic Regression model on a sample
try:
    predictions_lr = logistic_model.transform(spark_df_sample)
    predictions_lr = predictions_lr.select("id", "prediction").withColumnRenamed("prediction", "prediction_lr")
    print("Logistic Regression predictions successful on sample data.")
    predictions_lr.show()
except Exception as e:
    print(f"Error during Logistic Regression prediction: {e}")
    sys.exit(1)

# Perform prediction with Random Forest model on a sample
try:
    predictions_rf = rf_model.transform(spark_df_sample)
    predictions_rf = predictions_rf.select("id", "prediction").withColumnRenamed("prediction", "prediction_rf")
    print("Random Forest predictions successful on sample data.")
    predictions_rf.show()
except Exception as e:
    print(f"Error during Random Forest prediction: {e}")
    sys.exit(1)

# Join predictions and save as CSV
try:
    predictions = predictions_lr.join(predictions_rf, on="id")
    predictions.write.csv("fraud_detection_project/output/fraud_predictions_sample.csv", header=True, mode="overwrite")
    print("Sample predictions saved to fraud_detection_project/output/fraud_predictions_sample.csv")
except Exception as e:
    print(f"Error saving predictions: {e}")


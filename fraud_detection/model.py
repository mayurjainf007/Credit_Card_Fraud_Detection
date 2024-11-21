from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler as SklearnScaler
import pickle

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionModelTraining").getOrCreate()

# Load and preprocess dataset
data_path = "anydata.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Define label and features
label_col = "Class"
feature_cols = [col for col in df.columns if col != label_col]
df = df.withColumn(label_col, col(label_col).cast(DoubleType()))

# Assemble and scale features (PySpark)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
df = assembler.transform(df)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression Model (PySpark)
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol=label_col)
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol=label_col)
cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

cv_model = cv.fit(train_data)
cv_model.bestModel.write().overwrite().save("models/pyspark_logistic_regression")
scaler_model.write().overwrite().save("models/pyspark_scaler")

print("PySpark model and scaler saved.")

# Train a Scikit-Learn model and save as .pkl
pandas_df = df.toPandas()
X = pandas_df[feature_cols]
y = pandas_df[label_col]

# Standardize features (Scikit-Learn)
sklearn_scaler = SklearnScaler()
X_scaled = sklearn_scaler.fit_transform(X)

# Logistic Regression (Scikit-Learn)
sklearn_lr = SklearnLR()
sklearn_lr.fit(X_scaled, y)

# Save Scikit-Learn model
with open("models/fraud_detection_model.pkl", "wb") as model_file:
    pickle.dump((sklearn_lr, sklearn_scaler), model_file)

print("Scikit-Learn model saved.")


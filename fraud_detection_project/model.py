
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionModelTraining").getOrCreate()

# Load the dataset and infer schema
data = spark.read.csv("fraud_detection_project/dataset/fraudTrain1.csv", header=True, inferSchema=True)

# Define label column and exclude non-numeric columns
label_col = "is_fraud"
feature_cols = [c for c in data.columns if c not in ["id", "is_fraud", "first", "last", "street", "city", "state", "job", "dob", "trans_num"]]

# Cast feature columns to DoubleType and handle nulls
for col_name in feature_cols:
    data = data.withColumn(col_name, col(col_name).cast(DoubleType()))
data = data.na.fill(0, subset=feature_cols)

# Create weight column to handle class imbalance
fraud_count = data.filter(data.is_fraud == 1).count()
non_fraud_count = data.filter(data.is_fraud == 0).count()
fraud_weight = non_fraud_count / fraud_count
data = data.withColumn("classWeight", when(col("is_fraud") == 1, fraud_weight).otherwise(1.0))

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(data)
# Save scaler model
scaler_model.write().overwrite().save("fraud_detection_project/model/scalermodel")
data = scaler_model.transform(data)

# Split the data into train and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=11)

# Define evaluators
binary_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="prediction", metricName="areaUnderROC")

# Logistic Regression Model with Cross-Validation
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol=label_col, weightCol="classWeight")
paramGrid_lr = ParamGridBuilder()    .addGrid(lr.regParam, [0.01, 0.1, 0.5])    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])    .build()
crossval_lr = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid_lr, evaluator=binary_evaluator, numFolds=5)

# Random Forest Model with Cross-Validation
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol=label_col, weightCol="classWeight")
paramGrid_rf = ParamGridBuilder()    .addGrid(rf.numTrees, [50, 100])    .addGrid(rf.maxDepth, [5, 10])    .build()
crossval_rf = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid_rf, evaluator=binary_evaluator, numFolds=5)

# Train models with cross-validation
cvModel_lr = crossval_lr.fit(train_data)
cvModel_rf = crossval_rf.fit(train_data)

# Save models to specified directory with overwrite
cvModel_lr.bestModel.write().overwrite().save("fraud_detection_project/model/logisticregression")
cvModel_rf.bestModel.write().overwrite().save("fraud_detection_project/model/randomforest")

# Evaluate models on test set and generate additional metrics
predictions_lr = cvModel_lr.transform(test_data)
predictions_rf = cvModel_rf.transform(test_data)

# Collect ROC AUC and Accuracy
roc_auc_lr = binary_evaluator.evaluate(predictions_lr)
roc_auc_rf = binary_evaluator.evaluate(predictions_rf)

accuracy_lr = predictions_lr.filter(predictions_lr['prediction'] == predictions_lr[label_col]).count() / predictions_lr.count()
accuracy_rf = predictions_rf.filter(predictions_rf['prediction'] == predictions_rf[label_col]).count() / predictions_rf.count()

# Collect results
results = {
    "Model": ["Logistic Regression", "Random Forest"],
    "ROC_AUC": [roc_auc_lr, roc_auc_rf],
    "Accuracy": [accuracy_lr, accuracy_rf]
}
results_df = pd.DataFrame(results)

# Create output directory if it doesn't exist
output_dir = "fraud_detection_project/static/output"
os.makedirs(output_dir, exist_ok=True)

# Save results to CSV file
results_df.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)

# Plot ROC-AUC scores
plt.figure(figsize=(8, 6))
sns.barplot(data=results_df, x="Model", y="ROC_AUC", palette="viridis", legend=False)
plt.title("Model Comparison by ROC-AUC")
plt.savefig(os.path.join(output_dir, "model_comparison_plot.png"))
plt.close()

# Plot Accuracy scores
plt.figure(figsize=(8, 6))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="Blues", legend=False)
plt.title("Model Comparison by Accuracy")
plt.savefig(os.path.join(output_dir, "model_accuracy_plot.png"))
plt.close()

# Generate additional evaluation plots (ROC Curves and Confusion Matrices)
def plot_roc_curve(predictions, title, path):
    pdf = predictions.select(label_col, "probability").toPandas()
    y_true = pdf[label_col].values
    y_score = pdf["probability"].apply(lambda x: x[1]).values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()

plot_roc_curve(predictions_lr, "ROC Curve - Logistic Regression", os.path.join(output_dir, "roc_curve_lr.png"))
plot_roc_curve(predictions_rf, "ROC Curve - Random Forest", os.path.join(output_dir, "roc_curve_rf.png"))

# Confusion Matrix for each model
def plot_confusion_matrix(predictions, title, path):
    pdf = predictions.select(label_col, "prediction").toPandas()
    y_true = pdf[label_col].values
    y_pred = pdf["prediction"].values
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(path)
    plt.close()

plot_confusion_matrix(predictions_lr, "Confusion Matrix - Logistic Regression", os.path.join(output_dir, "confusion_matrix_lr.png"))
plot_confusion_matrix(predictions_rf, "Confusion Matrix - Random Forest", os.path.join(output_dir, "confusion_matrix_rf.png"))

print("Results, metrics, scaler model, confusion matrices, and additional graphs saved in output folder")

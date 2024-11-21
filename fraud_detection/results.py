import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import os

DATA_FOLDER = "fraud_detection/data"
OUTPUT_FOLDER = "fraud_detection/output"
STATIC_FOLDER = "fraud_detection/static"
MODEL_FOLDER = "fraud_detection/models"

def run_report():
    # Load the pre-trained model
    with open(os.path.join(MODEL_FOLDER, "fraud_detection_model.pkl"), "rb") as model_file:
        lr_model = pickle.load(model_file)

    # Load dataset
    df = pd.read_csv(os.path.join(DATA_FOLDER,"creditcard.csv"))

    # Prepare data for evaluation
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Predict probabilities using the pre-trained model
    y_pred_proba = lr_model.predict_proba(X_scaled)[:, 1]

    # Calculate confusion matrix
    y_pred = lr_model.predict(X_scaled)
    conf_matrix = confusion_matrix(y, y_pred)

    # Evaluate the model
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Save ROC-AUC score
    with open(os.path.join(STATIC_FOLDER,"model_accuracy.txt"), "w") as accuracy_file:
        accuracy_file.write(f"ROC-AUC Score: {roc_auc:.4f}\n")

    # Amount Distribution
    df_sorted = df.sort_values(by="Amount", ascending=False)
    sns.histplot(data=df_sorted, x="Amount", hue="Class", kde=True)
    plt.ylim(0, 6000)
    plt.xlim(0, 500) 
    plt.title("Transaction Amount Distribution")
    plt.savefig(os.path.join(STATIC_FOLDER,"amount_distribution.png"))
    plt.clf()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(STATIC_FOLDER,"roc_curve.png"))
    plt.clf()

    # Feature Importance
    feature_importance = abs(lr_model.coef_[0])
    features = df.columns[:-1]  # Exclude target column
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Save feature importance to a file
    feature_df.to_csv(os.path.join(STATIC_FOLDER,"feature_importance.csv"), index=False)

    # Plot top features
    sns.barplot(x='Importance', y='Feature', data=feature_df.head(10))
    plt.title("Top 10 Features by Importance")
    plt.savefig(os.path.join(STATIC_FOLDER,"feature_importance.png"))
    plt.clf()

    # Anomaly Detection with Isolation Forest
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    isolation_forest.fit(X_scaled)

    # Predict anomalies
    df['Anomaly'] = isolation_forest.predict(X_scaled)
    df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

    # Save anomaly results
    df.to_csv(os.path.join(STATIC_FOLDER,"transactions_with_anomalies.csv"), index=False)

    # Plot anomaly counts
    sns.countplot(x='Anomaly', data=df)
    plt.title("Anomaly Detection Results")
    plt.savefig(os.path.join(STATIC_FOLDER,"anomaly_detection.png"))
    plt.clf()

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=lr_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(STATIC_FOLDER,"confusion_matrix.png"))
    plt.clf()

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os

DATA_FOLDER = "fraud_detection/data"
MODEL_FOLDER = "fraud_detection/models"

# Original Pandas-based Implementation
def run_original():
    # Load the dataset
    data_path = os.path.join(DATA_FOLDER, "creditcard.csv")  # Update with your actual path
    df = pd.read_csv(data_path)

    # Split into features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Perform GridSearch to optimize Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200]
    }
    lr = LogisticRegression()
    grid_search = GridSearchCV(lr, param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train_resampled)

    # Save the model
    with open(os.path.join(MODEL_FOLDER,"fraud_detection_model.pkl"), "wb") as model_file:
        pickle.dump(grid_search.best_estimator_, model_file)

   

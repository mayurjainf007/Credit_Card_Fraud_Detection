
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
import os

app = Flask(__name__)
fraudulent_transactions = []

# Load model results for output view
results_df = pd.read_csv("fraud_detection_project/static/output/model_results.csv")
plot_paths = {
    "roc_auc_plot": "fraud_detection_project/static/output/model_comparison_plot.png",
    "conf_matrix_lr": "fraud_detection_project/static/output/confusion_matrix_lr.png",
    "conf_matrix_rf": "fraud_detection_project/static/output/confusion_matrix_rf.png",
    "pr_curve_lr": "fraud_detection_project/static/output/precision_recall_curve_lr.png",
    "pr_curve_rf": "fraud_detection_project/static/output/precision_recall_curve_rf.png"
}


@app.route('/')   
def index():
    return render_template(
        'dashboard.html',
        transactions=fraudulent_transactions,
        results=results_df.to_dict(orient="records"),
        plot_paths=plot_paths
    )

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    transaction = request.get_json()
    if transaction.get("prediction") == 1:
        fraudulent_transactions.append(transaction)
    return jsonify({"status": "Transaction added"}), 200

@app.route('/download')
def download_csv():
    if fraudulent_transactions:
        df = pd.DataFrame(fraudulent_transactions)
        csv_data = df.to_csv(index=False)
        return send_file(
            io.BytesIO(csv_data.encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="fraudulent_transactions.csv"
        )
    return jsonify({"error": "No transactions to download"}), 400

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    return jsonify(fraudulent_transactions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import subprocess
import threading
import time
from prediction import main as run_prediction # Import the prediction function
from producer import status

app = Flask(__name__)

# Paths
VENV_ACTIVATION = "source /home/rootx/Desktop/venv/bin/activate"
PRODUCER_SCRIPT = "fraud_detection/producer.py"
CONSUMER_SCRIPT = "fraud_detection/consumer.py"
DATA_FOLDER = "fraud_detection/data"
OUTPUT_FOLDER = "fraud_detection/output"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "fraudulent_transactions.csv")
FLAG = False

def run_producer(file_name):
    """Run the Kafka producer in a new terminal."""
    try:
        command = f"""gnome-terminal -- bash -c "{VENV_ACTIVATION} && sleep 15 && python {PRODUCER_SCRIPT} {file_name}; exec bash" """
        subprocess.Popen(command, shell=True)
    except Exception as e:
        print(f"Error running producer: {e}")

def run_consumer():
    """Run the Spark consumer in a new terminal."""
    try:
        command = f"""gnome-terminal -- bash -c "{VENV_ACTIVATION} && spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 {CONSUMER_SCRIPT}; exec bash" """
        subprocess.Popen(command, shell=True)
    except Exception as e:
        print(f"Error running consumer: {e}")

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_data():
    file = request.files['file']
    os.makedirs(DATA_FOLDER, exist_ok=True)  # Create data folder if it doesn't exist
    
    # Delete the existing file if it exists
    if os.path.isfile(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Deleted existing file: {OUTPUT_FILE}")

    if file:
        command = [
            "spark-submit",
            "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2",
            'fraud_detection/prediction.py',
            file.filename
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # Start Kafka producer and consumer in separate terminals
        producer_thread = threading.Thread(target=run_producer, args=(file.filename,))
        consumer_thread = threading.Thread(target=run_consumer)
        producer_thread.start()
        consumer_thread.start()
        producer_thread.join()  # Wait for producer to finish
        consumer_thread.join()  # Wait for consumer to finish
        
        return jsonify({"status": "success", "message": "Upload successful!!!"}), 200
    return jsonify({"status": "error", "message": "No file provided"}), 400

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Fetch fraudulent transactions."""
    if not os.path.isfile(OUTPUT_FILE):
        time.sleep(30)
    transactions = pd.read_csv(OUTPUT_FILE).to_dict(orient="records")
    return jsonify(transactions)

@app.route('/download')
def download_csv():
    """Download fraudulent transactions CSV."""
    output_path = os.path.join("output", "fraudulent_transactions.csv")
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True, download_name="fraudulent_transactions.csv")
    return jsonify({"error": "No fraudulent transactions to download"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


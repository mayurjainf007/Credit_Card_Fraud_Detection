from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io

app = Flask(__name__)
fraudulent_transactions = []

@app.route('/')
def index():
    return render_template('dashboard.html', transactions=fraudulent_transactions)

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    transaction = request.get_json()
    fraudulent_transactions.append(transaction)
    return jsonify({"status": "Transaction added"}), 200

@app.route('/download')
def download_csv():
    df = pd.DataFrame(fraudulent_transactions)
    csv_data = df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="fraudulent_transactions.csv"  # Updated argument name
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


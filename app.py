




import os
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, jsonify
from core.data_loader import load_data
from core.model_trainer import train_models
from core.predictor import generate_predictions

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FEATURE_COLS = [
    "Amount", "CountryRisk", "TransactionType", "CustomerType", "AccountAgeMonths",
    "HasPriorSAR", "RelatedPartiesCount", "TransactionLocation", "PurposeKnown",
    "TransactionPattern", "NarrativeRiskTerms"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", show_results=False)

@app.route("/predict", methods=["POST"])
def predict():
    # Automatically pick training and test file from folders
    train_files = os.listdir(os.path.join(UPLOAD_DIR, "train"))
    test_files = os.listdir(os.path.join(UPLOAD_DIR, "test"))

    if not train_files or not test_files:
        return "Training and test files not found in uploads directory.", 400

    train_path = os.path.join(UPLOAD_DIR, "train", train_files[0])
    test_path = os.path.join(UPLOAD_DIR, "test", test_files[0])

    train_df, test_df, train_encoded, test_encoded = load_data(train_path, test_path, FEATURE_COLS)
    trained_models, model_scores = train_models(train_encoded, train_df["IsFraud"])
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = trained_models[best_model_name]

    result_df = generate_predictions(best_model, best_model_name, test_encoded, test_df)

    # Save to results directory
    json_path = os.path.join(RESULTS_DIR, "json", "predictions.json")
    excel_path = os.path.join(RESULTS_DIR, "excel", "predictions.xlsx")
    result_df.to_json(json_path, orient="records", indent=2)
    result_df.to_excel(excel_path, index=False)

    # Convert to HTML table
    result_html = result_df.to_html(classes="table table-striped", index=False)

    return render_template("index.html", show_results=True, table=result_html)

@app.route("/download/json")
def download_json():
    path = os.path.join(RESULTS_DIR, "json")
    return send_from_directory(path, "predictions.json", as_attachment=True)

@app.route("/download/excel")
def download_excel():
    path = os.path.join(RESULTS_DIR, "excel")
    return send_from_directory(path, "predictions.xlsx", as_attachment=True)

if __name__ == "__main__":
     app.run(debug=True, host='0.0.0.0', port=8088)

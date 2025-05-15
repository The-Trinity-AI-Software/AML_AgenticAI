




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
    if "train_file" not in request.files or "test_file" not in request.files:
        return "Please upload both training and test files.", 400

    train_file = request.files["train_file"]
    test_file = request.files["test_file"]

    if not train_file or not test_file:
        return "Empty files received.", 400

    # Save uploaded files to a temporary location
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    train_path = os.path.join(UPLOAD_DIR, "uploaded_train.csv")
    test_path = os.path.join(UPLOAD_DIR, "uploaded_test.csv")
    train_file.save(train_path)
    test_file.save(test_path)

    # Load data, train models, predict
    train_df, test_df, train_encoded, test_encoded = load_data(train_path, test_path, FEATURE_COLS)
    trained_models, model_scores = train_models(train_encoded, train_df["IsFraud"])
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = trained_models[best_model_name]
    result_df = generate_predictions(best_model, best_model_name, test_encoded, test_df)

    # Save results
    os.makedirs(os.path.join(RESULTS_DIR, "json"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "excel"), exist_ok=True)
    result_df.to_json(os.path.join(RESULTS_DIR, "json", "predictions.json"), orient="records", indent=2)
    result_df.to_excel(os.path.join(RESULTS_DIR, "excel", "predictions.xlsx"), index=False)
    os.remove(train_path)
    os.remove(test_path)
    
    result_html = result_df.to_html(classes="table table-striped", index=False)
    
    return render_template("index.html", show_results=True, table=result_html)
    
@app.route("/routes")
def show_routes():
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        output.append(f"{rule.endpoint}: {rule.rule} [{methods}]")
    return "<br>".join(sorted(output))

@app.route("/download/json")
def download_json():
    path = os.path.join(RESULTS_DIR, "json")
    return send_from_directory(path, "predictions.json", as_attachment=True)

@app.route("/download/excel")
def download_excel():
    path = os.path.join(RESULTS_DIR, "excel")
    return send_from_directory(path, "predictions.xlsx", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=7002)

🛡️ AML Fraud Detection System (Flask + LangGraph + ML Models)

This project is a lightweight Flask web app for Anti Money Laundering (AML) fraud detection using a hybrid Agentic AI pipeline.It combines:

📋 Rule-based fraud indicators (amount, location, transaction type)

🤖 ML-based risk scoring (XGBoost, Random Forest, Logistic Regression)

🧐 Explainable AI using LangGraph-style agent workflows

📂 Project Structure

.
├── app.py                # Flask app backend
├── core/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── model_trainer.py  # Model training (XGB, RF, LR)
│   └── predictor.py      # LangGraph pipeline for prediction and explanation
├── templates/
│   └── index.html        # Front-end UI
├── uploads/
│   ├── train/
│   │   └── aml_rich_sample_dataset.csv  # Example rich training dataset
│   └── test/
│       ├── aml_test_for_prediction.csv  # Example test dataset (for prediction)
│       └── aml_test_with_target.csv     # Example test dataset (with fraud target labels)
├── results/
│   ├── excel/            # Prediction Excel outputs
│   └── json/             # Prediction JSON outputs
├── README.md             # (This file)

🚀 How It Works

Upload DataUpload a training CSV and a test CSV via the UI.

Preprocessing

Automatically selects key AML features:

["Amount", "CountryRisk", "TransactionType", "CustomerType", "AccountAgeMonths",
 "HasPriorSAR", "RelatedPartiesCount", "TransactionLocation", "PurposeKnown",
 "TransactionPattern", "NarrativeRiskTerms"]

Encodes categorical columns using One-Hot Encoding (pd.get_dummies).

Model Training

Trains 3 models:

XGBoost

Random Forest

Logistic Regression

Applies SMOTE oversampling to balance the dataset.

Selects the best model based on ROC-AUC Score.

Agentic Fraud Detection (LangGraph pipeline)

Node 1: Rule-based risk checks (high value, high risk country, etc.)

Node 2: ML-based prediction (fraudulent or legitimate)

Node 3: Explainable AI generation (reason for prediction)

Results Download

Download Excel and JSON results.

View Top 15 Predictions directly on the dashboard.

📦 Example Datasets

Folder

File

Purpose

uploads/train/

aml_rich_sample_dataset.csv

Rich training dataset

uploads/test/

aml_test_for_prediction.csv

Test dataset for predictions

uploads/test/

aml_test_with_target.csv

Test dataset (with actual fraud labels)

📸 Dashboard UI (Features)

📄 Upload Train and Test files easily

🏆 Run prediction instantly

📊 View top 15 prediction results interactively

📥 Download full results in Excel / JSON

🧐 Readable explanations for each prediction

🧹 Future Enhancements

Model hyperparameter tuning via UI

Add multiple model evaluation

Full API version for production deployments

User authentication for upload/downloads

Azure/AWS cloud deployment option


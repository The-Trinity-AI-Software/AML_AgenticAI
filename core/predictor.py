import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AMLState(TypedDict):
    transaction: pd.Series
    rule_flag: str
    prediction: str
    best_model: str
    reason: str
    explanation: str
    action: str

def rule_check_node(state: AMLState) -> AMLState:
    tx = state["transaction"]
    flags = []
    if tx.get("Amount", 0) > 10000: flags.append("High-value transaction")
    if tx.get("CountryRisk", 0) >= 8: flags.append("High-risk country")
    if tx.get("TransactionLocation_foreign", 0) == 1: flags.append("Foreign transaction")
    if tx.get("PurposeKnown", 1) == 0: flags.append("Unknown purpose")
    if tx.get("TransactionPattern_structuring", 0) == 1: flags.append("Possible structuring")
    if tx.get("HasPriorSAR", 0) == 1: flags.append("Prior SAR exists")
    state["rule_flag"] = ", ".join(flags) if flags else "No rule triggers"
    return state

def model_predict_node(state: AMLState, best_model, best_model_name) -> AMLState:
    tx = state["transaction"]
    prob = best_model.predict_proba(tx.values.reshape(1, -1))[0][1]
    label = "Suspicious" if prob > 0.3 else "Non-suspicious"

    rules = state["rule_flag"].split(", ") if state["rule_flag"] != "No rule triggers" else []
    reasons = []
    if "High-value transaction" in rules: reasons.append("amount exceeds $10,000")
    if "High-risk country" in rules: reasons.append("transaction involves a high-risk country")
    if "Foreign transaction" in rules: reasons.append("foreign location")
    if "Unknown purpose" in rules: reasons.append("missing declared purpose")
    if "Prior SAR exists" in rules: reasons.append("previous suspicious activity reported")
    if "Possible structuring" in rules: reasons.append("structuring pattern detected")

    if label == "Suspicious":
        explanation = "This transaction is considered suspicious because it " + ", ".join(reasons) + "." if reasons else "Flagged by model due to risk score."
    else:
        explanation = "Although rule-based indicators are present (" + ", ".join(reasons) + "), the model considers this transaction non suspicious with confidence." if reasons else "No risk indicators detected."

    action = (
        "Suspicious Activity Report (SAR) is required to be filed with FinCEN. Task to be handed over to SAR_Filing_Agent"
        if label == "Suspicious" else "No Action"
    )

    state["prediction"] = label
    state["best_model"] = best_model_name
    state["reason"] = f"{label} (Confidence: {prob:.2f})"
    state["explanation"] = explanation
    state["action"] = action
    return state

def generate_predictions(best_model, best_model_name, test_encoded, test_df):
    graph = StateGraph(AMLState)
    graph.add_node("RuleCheck", rule_check_node)
    graph.add_node("ModelPredict", lambda state: model_predict_node(state, best_model, best_model_name))
    graph.set_entry_point("RuleCheck")
    graph.add_edge("RuleCheck", "ModelPredict")
    graph.add_edge("ModelPredict", END)
    pipeline = graph.compile()

    results = []
    for i in range(len(test_encoded)):
        row = test_encoded.iloc[i]
        result = pipeline.invoke({"transaction": row})
        results.append({
            "TransactionID": test_df.iloc[test_encoded.index[i]]["TransactionID"],
            "Prediction": result["prediction"],
            "Confidence": result["reason"].split("Confidence: ")[-1].replace(")", ""),
            "Model": result["best_model"],
            "Explanation": result["explanation"],
            "Action": result["action"]
        })
    return pd.DataFrame(results)

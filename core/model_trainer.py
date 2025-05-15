from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

def train_models(X_train, y_train):
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    models = {
        "XGB": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "RF": RandomForestClassifier(n_estimators=100),
        "LR": LogisticRegression(max_iter=1000)
    }
    scores, trained = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_train)[:, 1]
        scores[name] = roc_auc_score(y_train, prob)
        trained[name] = model
    return trained, scores

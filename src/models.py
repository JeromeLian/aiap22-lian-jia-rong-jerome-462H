
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def build_models() -> Dict[str, object]:
    """
    Return a dict of candidate models.
    You can tweak hyperparameters based on experiments.
    """
    models = {
        "log_reg": LogisticRegression(
            max_iter=5000,          # <-- this is where you increase max_iter
            class_weight="balanced" # helps with class imbalance
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        ),
        "grad_boost": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }
    return models

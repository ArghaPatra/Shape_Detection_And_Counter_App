import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MINIMAL = ["n_vertices_4", "circle_fill", "rect_fill", "rect_aspect", "hu1"]

feats = pd.read_csv("data/processed/features.csv")
model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=0).fit(feats[MINIMAL], feats["label"])

os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "features": MINIMAL}, "models/shape_rf.joblib")
print("saved models/shape_rf.joblib, trained on", len(feats), "images")
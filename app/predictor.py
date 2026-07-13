import joblib
import os

# Prefer the artifact saved by the training pipeline. Fall back to models/ for compatibility.
MODEL_PATHS = [
	os.path.join("artifacts", "churn_pipeline.pkl"),
	os.path.join("models", "churn_pipeline.pkl"),
]

for path in MODEL_PATHS:
	if os.path.exists(path):
		pipeline = joblib.load(path)
		break
else:
	raise FileNotFoundError(
		"Could not find churn_pipeline.pkl. Expected one of: {}".format(MODEL_PATHS)
	)
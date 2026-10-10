import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Adjust these paths if your datasets are stored elsewhere.
DATA_DIR = ROOT / "data" / "raw"
FEATURES_FILE = ROOT / "docs" / "validated_selected_features.csv"

# Load the original IEEE-CIS datasets.
transactions = pd.read_csv(DATA_DIR / "train_transaction.csv")
identity = pd.read_csv(DATA_DIR / "train_identity.csv")

df = transactions.merge(
    identity,
    on="TransactionID",
    how="left"
)

# Read the validated feature list.
feature_df = pd.read_csv(FEATURES_FILE)

print("Feature-list columns:", feature_df.columns.tolist())

# Update this selection if your CSV uses a different column name.
possible_names = ["feature", "feature_name", "selected_feature"]
feature_col = next(
    (c for c in possible_names if c in feature_df.columns),
    None
)

if feature_col is None:
    raise ValueError(
        "Inspect the feature-list CSV and identify its feature-name column."
    )

features = feature_df[feature_col].dropna().astype(str).tolist()

assert len(features) == 80, f"Expected 80 features, found {len(features)}"
assert len(set(features)) == 80, "Duplicate feature names detected"

missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Features missing from merged data: {missing}")

# Choose a real transaction for an initial smoke test.
row = df.iloc[0]

payload = {}
for feature in features:
    value = row[feature]

    if pd.isna(value):
        payload[feature] = None
    elif hasattr(value, "item"):
        payload[feature] = value.item()
    else:
        payload[feature] = value

output = ROOT / "sample_transaction.json"
output.write_text(
    json.dumps(payload, indent=2, allow_nan=False),
    encoding="utf-8"
)

print(f"Selected features: {len(payload)}")
print(f"Payload saved to: {output}")
print(json.dumps(payload, indent=2, allow_nan=False))
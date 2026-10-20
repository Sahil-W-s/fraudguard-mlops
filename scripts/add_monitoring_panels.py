
import json
from pathlib import Path

dashboard_path = Path("monitoring/dashboards/fraudguard-monitoring.json")

with dashboard_path.open("r", encoding="utf-8") as file:
    dashboard = json.load(file)

panels = dashboard["panels"]
existing_ids = {panel["id"] for panel in panels}
existing_titles = {panel["title"] for panel in panels}

# Refuse to proceed if this script has already added any of the panels.
new_titles = {
    "Labeled Predictions",
    "Actual Fraud Rate",
    "Evaluation Precision",
    "Evaluation Recall",
}
if existing_titles & new_titles:
    raise SystemExit(
        "At least one evaluation panel already exists. "
        "Inspect the dashboard before running this script again."
    )

# Keep the working zero-display behavior for the HTTP 5xx panel.
for panel in panels:
    if panel.get("title") == "HTTP 5xx Error Rate (%)":
        for target in panel.get("targets", []):
            if "fraudguard_http_requests_total" in target.get("expr", ""):
                target["expr"] = (
                    '(100 * sum(rate(fraudguard_http_requests_total'
                    '{status_code=~"5.."}[5m])) / '
                    'clamp_min(sum(rate(fraudguard_http_requests_total[5m])), '
                    '1e-9)) or vector(0)'
                )

def make_stat_panel(
    panel_id, title, expression, x, y, unit="percentunit",
    no_value=None, maximum=1
):
    defaults = {
        "unit": unit,
        "min": 0,
    }
    if maximum is not None:
        defaults["max"] = maximum
    if no_value is not None:
        defaults["noValue"] = no_value

    return {
        "id": panel_id,
        "title": title,
        "type": "stat",
        "gridPos": {"h": 5, "w": 6, "x": x, "y": y},
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "targets": [{"expr": expression, "refId": "A"}],
        "fieldConfig": {
            "defaults": defaults,
            "overrides": [],
        },
        "options": {
            "reduceOptions": {
                "calcs": ["lastNotNull"],
                "fields": "",
                "values": False,
            },
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto",
            "wideLayout": True,
        },
    }

available_ids = set(range(1, 100))
available_ids -= existing_ids
if len(available_ids) < 4:
    raise SystemExit("Could not find four available panel IDs.")

ids = sorted(available_ids)[:4]

new_panels = [
    make_stat_panel(
        ids[0], "Labeled Predictions",
        "fraudguard_labeled_predictions", 0, 24,
        unit="short", maximum=None,
    ),
    make_stat_panel(
        ids[1], "Actual Fraud Rate",
        "fraudguard_actual_fraud_rate", 6, 24,
    ),
    make_stat_panel(
        ids[2], "Evaluation Precision",
        "fraudguard_evaluation_precision", 12, 24,
        no_value="No value",
    ),
    make_stat_panel(
        ids[3], "Evaluation Recall",
        "fraudguard_evaluation_recall", 0, 29,
    ),
]

dashboard["panels"].extend(new_panels)

# Save a backup before changing the original file.
backup_path = dashboard_path.with_suffix(".json.bak")
backup_path.write_text(
    json.dumps(json.loads(dashboard_path.read_text(encoding="utf-8")), indent=2)
    + "\n",
    encoding="utf-8",
)

dashboard_path.write_text(
    json.dumps(dashboard, indent=2) + "\n",
    encoding="utf-8",
)

# Validate the written JSON and panel count.
with dashboard_path.open("r", encoding="utf-8") as file:
    verified = json.load(file)

assert len(verified["panels"]) == len(panels) + 4
print(f"Updated dashboard: {len(verified['panels'])} panels.")
print(f"Backup created at: {backup_path}")
print("Existing panels preserved; four evaluation panels added.")

from __future__ import annotations

import argparse
import html
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from scripts import run_experiments as experiments

ROOT = Path(__file__).resolve().parents[1]
REQUESTS_PATH = ROOT / "data" / "monitoring" / "inference_requests.csv"
REPORT_DIR = ROOT / "monitoring" / "drift_reports"

PSI_THRESHOLD = float(
    os.getenv("FRAUDGUARD_PSI_THRESHOLD", "0.20")
)
KS_PVALUE_THRESHOLD = float(
    os.getenv("FRAUDGUARD_KS_PVALUE", "0.05")
)
MIN_ROWS = int(os.getenv("FRAUDGUARD_DRIFT_MIN_ROWS", "100"))
DRIFT_FEATURE_FRACTION = float(
    os.getenv("FRAUDGUARD_DRIFT_FEATURE_FRACTION", "0.20")
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI using reference quantile bins."""

    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if ref.empty or cur.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    if len(edges) < 2:
        return 0.0 if np.allclose(
            cur.to_numpy(), ref.median()
        ) else float("inf")

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)

    # Smoothing prevents division by zero.
    ref_pct = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_pct = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    max_categories: int = 30,
) -> float:
    """Calculate categorical PSI, grouping rare categories as OTHER."""

    ref = reference.astype("string").fillna("__MISSING__")
    cur = current.astype("string").fillna("__MISSING__")

    top = ref.value_counts().head(max_categories).index.tolist()
    allowed = set(top) | {"__MISSING__", "__OTHER__"}

    ref = ref.where(ref.isin(allowed), "__OTHER__")
    cur = cur.where(cur.isin(allowed), "__OTHER__")

    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy(float)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy(float)

    ref_pct = (ref_counts + 0.5) / (len(ref) + 0.5 * len(categories))
    cur_pct = (cur_counts + 0.5) / (len(cur) + 0.5 * len(categories))

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def load_reference() -> tuple[pd.DataFrame, list[str]]:
    """Load the existing chronological training reference."""

    data = experiments.load_splits()
    feature_names = list(data["feature_pipeline"].feature_names)
    reference = data["train_df"][feature_names].copy()

    if reference.empty:
        raise ValueError("Training reference dataset is empty.")

    return reference, feature_names


def generate_demo_batch(
    reference: pd.DataFrame,
    feature_names: list[str],
) -> Path:
    """Generate a synthetic batch with deliberately widespread numeric drift."""
    rng = np.random.default_rng(42)

    sample_size = min(max(MIN_ROWS, 500), len(reference))
    demo = reference[feature_names].sample(
        n=sample_size,
        random_state=42,
    ).copy()

    numeric_features = [
        name
        for name in feature_names
        if pd.api.types.is_numeric_dtype(reference[name])
    ]

    # Shift enough features to exceed the configured overall threshold.
    required_count = int(
        np.ceil(DRIFT_FEATURE_FRACTION * len(feature_names))
    )
    shift_count = min(
        len(numeric_features),
        max(required_count + 4, 20),
    )

    if shift_count < required_count:
        raise ValueError(
            f"Cannot simulate sufficient drift: only "
            f"{len(numeric_features)} numeric features available."
        )

    for feature in numeric_features[:shift_count]:
        values = pd.to_numeric(demo[feature], errors="coerce")
        valid = values.dropna()

        scale = float(valid.std()) if len(valid) > 1 else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        # Preserve missing values while changing the feature distribution.
        demo[feature] = values * 3.0 + (3.0 * scale)

    REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(REQUESTS_PATH, index=False)

    logger.warning(
        "Synthetic demonstration batch written to %s. "
        "Shifted %d of %d numeric features; this is not real production traffic.",
        REQUESTS_PATH,
        shift_count,
        len(numeric_features),
    )

    return REQUESTS_PATH


def run_drift_check() -> dict:
    """Compare recent inference requests against training features."""

    if not 0 < DRIFT_FEATURE_FRACTION <= 1:
        raise ValueError("DRIFT_FEATURE_FRACTION must be in (0, 1].")
    if MIN_ROWS < 2:
        raise ValueError("Minimum drift sample size must be at least 2.")
    if PSI_THRESHOLD < 0 or not 0 < KS_PVALUE_THRESHOLD < 1:
        raise ValueError("Invalid PSI or KS threshold.")

    reference, feature_names = load_reference()

    if not REQUESTS_PATH.is_file():
        return {
            "status": "insufficient_data",
            "drift_detected": False,
            "reason": f"Recent request file not found: {REQUESTS_PATH}",
            "report_path": None,
        }

    current = pd.read_csv(REQUESTS_PATH)

    missing = sorted(set(feature_names) - set(current.columns))
    if missing:
        raise ValueError(
            f"Inference request file is missing selected features: {missing}"
        )

    current = current[feature_names]

    if len(reference) < MIN_ROWS or len(current) < MIN_ROWS:
        return {
            "status": "insufficient_data",
            "drift_detected": False,
            "reference_rows": len(reference),
            "current_rows": len(current),
            "minimum_rows": MIN_ROWS,
            "report_path": None,
        }

    results = []

    for feature in feature_names:
        ref = reference[feature]
        cur = current[feature]

        numeric = (
            pd.api.types.is_numeric_dtype(ref)
            and pd.api.types.is_numeric_dtype(cur)
        )

        if numeric:
            psi = population_stability_index(ref, cur)
            ref_values = pd.to_numeric(ref, errors="coerce").dropna()
            cur_values = pd.to_numeric(cur, errors="coerce").dropna()

            ks = (
                ks_2samp(ref_values, cur_values)
                if len(ref_values) and len(cur_values)
                else None
            )

            ks_statistic = float(ks.statistic) if ks else None
            ks_pvalue = float(ks.pvalue) if ks else None
        else:
            psi = categorical_psi(ref, cur)
            ks_statistic = None
            ks_pvalue = None

        results.append({
            "feature": feature,
            "type": "numeric" if numeric else "categorical",
            "psi": psi if np.isfinite(psi) else None,
            "psi_infinite": not np.isfinite(psi),
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "drifted": (
                not np.isfinite(psi)
                or psi >= PSI_THRESHOLD
                or (
                    ks_pvalue is not None
                    and ks_pvalue < KS_PVALUE_THRESHOLD
                )
            ),
        })

    drifted_count = sum(item["drifted"] for item in results)
    drift_fraction = drifted_count / len(results) if results else 0.0

    detected = drift_fraction >= DRIFT_FEATURE_FRACTION

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "drift_detected" if detected else "no_drift_detected",
        "drift_detected": detected,
        "reference_source": "chronological training split",
        "current_source": str(REQUESTS_PATH.relative_to(ROOT)),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "feature_count": len(results),
        "drifted_feature_count": drifted_count,
        "drifted_feature_fraction": drift_fraction,
        "thresholds": {
            "psi": PSI_THRESHOLD,
            "ks_pvalue": KS_PVALUE_THRESHOLD,
            "drifted_feature_fraction": DRIFT_FEATURE_FRACTION,
            "minimum_rows": MIN_ROWS,
        },
        "features": results,
    }

    json_path = REPORT_DIR / f"drift_{timestamp}.json"
    html_path = REPORT_DIR / f"drift_{timestamp}.html"
    latest_json = REPORT_DIR / "latest.json"
    latest_html = REPORT_DIR / "latest.html"

    report["report_path"] = str(json_path.relative_to(ROOT))
    report["html_report_path"] = str(html_path.relative_to(ROOT))

    payload = json.dumps(report, indent=2, allow_nan=False)
    json_path.write_text(payload, encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")

    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(item["feature"]),
            html.escape(item["type"]),
            html.escape(str(item.get("psi", "N/A"))),
            "Yes" if item.get("drifted") else "No",
        )
        for item in sorted(
            results,
            key=lambda item: (
                not item.get("drifted", False),
                item["feature"],
            ),
        )
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FraudGuard Drift Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;color:#222}}
table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px;border:1px solid #ddd;text-align:left}}
th{{background:#eee}} .status{{font-weight:bold}}
</style></head><body>
<h1>FraudGuard Drift Report</h1>
<p class="status">Decision: {html.escape(report["status"])}</p>
<p>Reference rows: {len(reference)} | Current rows: {len(current)}</p>
<p>Drifted features: {drifted_count}/{len(results)}</p>
<p>Generated UTC: {html.escape(report["timestamp_utc"])}</p>
<table><thead><tr>
<th>Feature</th><th>Type</th><th>PSI</th><th>Drifted</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""

    html_path.write_text(page, encoding="utf-8")
    latest_html.write_text(page, encoding="utf-8")

    report["report_path"] = str(json_path.relative_to(ROOT))
    report["html_report_path"] = str(html_path.relative_to(ROOT))

    logger.info(
        "Drift check completed: detected=%s, drifted_features=%s/%s, report=%s",
        detected, drifted_count, len(results), json_path,
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Create a synthetic shifted batch before checking drift.",
    )
    args = parser.parse_args()

    if args.simulate_drift:
        reference, feature_names = load_reference()
        generate_demo_batch(reference, feature_names)

    result = run_drift_check()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

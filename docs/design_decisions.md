\# Design Decisions



\## Phase 2.2 — Feature Selection



\### Status



Completed initial feature selection and chronological evaluation. The 80-feature set is the preferred candidate for subsequent development.



\### 1. Dataset and feature inventory



\- Dataset: IEEE-CIS Fraud Detection

\- Joined transactions: 590,540

\- Initial candidate predictive features: 431

\- Selected candidate features: 80

\- Features removed from the candidate set: 351

\- Feature-count reduction: 81.44%



\### 2. Selection methodology



Feature selection used three sources of evidence:



1\. \*\*Missingness analysis:\*\* identified features with high percentages of missing values.

2\. \*\*Distribution analysis:\*\* identified low-cardinality and near-constant features.

3\. \*\*LightGBM gain importance:\*\* ranked features by their contribution to model training.



The initial 80 features captured 90.80% of the original model's cumulative gain importance. This is a ranking statistic, not a guarantee that the reduced model retains 90.80% of predictive performance.



The final shortlist was regenerated using the chronological training period.



\### 3. Chronological evaluation



The joined dataset was split by TransactionDT:



| Split | Rows | Purpose |

|---|---:|---|

| Training | 413,378 | Model fitting and feature ranking |

| Validation | 88,581 | Early stopping |

| Test | 88,581 | Final comparison |



The test period follows the training and validation periods chronologically.



\### 4. Evaluation results



| Metric | Full model | Reduced model |

|---|---:|---:|

| Features | 431 | 80 |

| Best iteration | 300 | 298 |

| ROC-AUC | 0.900742 | 0.900248 |

| Average precision | 0.515728 | 0.515217 |

| Log loss | 0.093546 | 0.093340 |



\### 5. Decision and rationale



The 80-feature model is the preferred candidate for the next development phase because it reduces the feature count by 81.44% while maintaining similar measured performance on the chronological test period.



Compared with the full model:



\- ROC-AUC decreases by 0.000494.

\- Average precision decreases by 0.000511.

\- Log loss improves by 0.000206.



These results support the reduced model as a practical candidate. They do not establish statistical equivalence or guarantee performance across future time periods.



\### 6. Limitations and follow-up work



\- Both models reached the configured 300-estimator limit or a nearby iteration. Further tuning may improve the comparison.

\- Evaluation covers one chronological split.

\- Confidence intervals and repeated temporal evaluations have not been performed.

\- The selected features and their preprocessing must be integrated consistently into training and inference.

\- Feature importance can vary with the model and dataset period.

\- The final production feature set should be re-evaluated after integration.



\### 7. Reproducibility artifacts



\- `docs/feature\_inventory.csv`

\- `docs/feature\_selection\_report.csv`

\- `docs/feature\_distribution\_profile.csv`

\- `docs/lightgbm\_feature\_importance.csv`

\- `docs/combined\_feature\_selection\_report.csv`

\- `docs/selected\_features.csv`

\- `docs/validated\_feature\_importance.csv`

\- `docs/validated\_selected\_features.csv`

\- `docs/feature\_selection\_evaluation.csv`

\- `scripts/run\_lightgbm\_selection.py`

\- `scripts/build\_feature\_selection\_report.py`

\- `scripts/create\_feature\_shortlist.py`

\- `scripts/evaluate\_feature\_shortlist.py`

## Phase 3: Experiment Tracking and Model Selection

### 1. Experiment Tracking with MLflow

**Decision:** Use MLflow to track model experiments, parameters, evaluation metrics, artifacts, and registered model versions.

**Rationale:** Centralized experiment tracking makes model comparisons reproducible and provides a record of model development.

### 2. Chronological Train/Validation/Test Split

**Decision:** Split the IEEE-CIS dataset chronologically into 70% training, 15% validation, and 15% testing.

**Rationale:** Fraud patterns change over time. A chronological split better approximates the task of predicting future transactions than a random split.

**Implementation:** Sort transactions by `TransactionDT` before splitting. Fit the feature pipeline using training data only.

### 3. Compare Three Models

**Decision:** Evaluate Logistic Regression, LightGBM, and Random Forest.

**Rationale:**
- **Logistic Regression:** Establishes a baseline and offers a simpler model for comparison.
- **LightGBM:** Captures nonlinear relationships and feature interactions efficiently.
- **Random Forest:** Provides an alternative tree-based model for comparison.

### 4. Select Models Using Average Precision

**Decision:** Select the winning model using validation average precision (AP).

**Rationale:** Fraud is a minority class, so accuracy can be misleading. Average precision evaluates the precision-recall trade-off across classification thresholds.

**Observed result:** LightGBM achieved the highest validation AP of `0.537374`, followed by Random Forest at `0.479839` and Logistic Regression at `0.367217`.

### 5. Choose a Decision Threshold Using a Precision Constraint

**Decision:** On the validation set, select the threshold that maximizes recall while maintaining precision of at least 0.80.

**Rationale:** The constraint aims to limit false-positive alerts while identifying as many fraudulent transactions as possible.

**Observed result:** The selected LightGBM threshold was approximately `0.9002603`, with validation precision of `0.800308` and recall of `0.341223`.

**Limitation:** Test precision was `0.744853`, below the validation target. The threshold therefore does not guarantee 80% precision on future data and requires further evaluation before deployment.

### 6. Register the Winning Model

**Decision:** Register the selected model as `FraudGuardFraudDetector` and assign the `champion` alias to version 1.

**Rationale:** A stable model name and alias provide a consistent reference for inference applications while allowing the underlying model version to change through controlled updates.

### 7. Preserve Model Lineage

**Decision:** Record the Git commit, worktree status, dataset DVC hashes, feature-list path, experiment run ID, model version, and evaluation metrics.

**Rationale:** Lineage helps trace a registered model back to its data and training experiment.

**Limitation:** The worktree was dirty during training. The recorded Git commit alone does not fully identify the exact source code used for the experiment.
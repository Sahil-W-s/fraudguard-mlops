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


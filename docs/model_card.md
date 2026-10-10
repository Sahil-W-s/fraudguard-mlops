# FraudGuard — Model Card

## 1. Model Overview

- **Project:** FraudGuard MLOps
- **Model:** LightGBM fraud classifier
- **Registered model:** `FraudGuardFraudDetector`
- **Registered version:** 1
- **Registry alias:** `champion`
- **Experiment:** `FraudGuard-Fraud-Detection`
- **Dataset:** IEEE-CIS Fraud Detection
- **Selected features:** 80
- **Selection criterion:** Validation average precision (AP)

## 2. Intended Use

The model estimates the likelihood that a transaction is fraudulent and supports transaction-risk prioritization.

Potential uses include fraud-review queues, transaction risk scoring, and evaluation of fraud-detection workflows.

This model has not been established as production-ready. Decisions involving customers or financial transactions should include appropriate operational controls and human oversight.

## 3. Training and Evaluation Methodology

The joined IEEE-CIS transaction and identity datasets contain 590,540 transaction rows.

The dataset was sorted chronologically using `TransactionDT` and split into:

| Partition | Rows | Proportion |
|---|---:|---:|
| Training | 413,378 | 70% |
| Validation | 88,581 | 15% |
| Test | 88,581 | 15% |

The feature pipeline was fitted using training data. Validation average precision determined model selection. The test set was reserved for final evaluation.

Training fraud prevalence was approximately 3.517%.

## 4. Model Selection

| Model | Validation AP | Recall at precision target |
|---|---:|---:|
| LightGBM | 0.537374 | 0.341223 |
| Random Forest | 0.479839 | 0.281394 |
| Logistic Regression | 0.367217 | 0.131492 |

LightGBM was selected because it achieved the highest validation average precision.

The validation decision threshold was approximately 0.9002603, selected to maximize recall subject to a target precision of at least 0.80.

## 5. Test Performance

| Metric | Test result |
|---|---:|
| Average precision | 0.489237 |
| ROC-AUC | 0.894692 |
| Log loss | 0.256519 |
| Precision | 0.744853 |
| Recall | 0.328576 |
| F1 score | 0.455998 |
| Accuracy | 0.972714 |

Test precision was below the validation target. The validation precision constraint therefore should not be interpreted as a guarantee of future precision.

Accuracy alone is insufficient for judging this model because fraud is a minority class.

## 6. Threshold and Operational Considerations

The threshold was selected using validation data, not the held-out test set. Its performance may change as transaction patterns and fraud prevalence change.

Before operational deployment:
- Evaluate precision and recall at realistic transaction volumes.
- Assess false-positive and false-negative costs.
- Establish monitoring for data drift and performance degradation.
- Define a process for reviewing flagged transactions.
- Revalidate threshold choices on an appropriate, independent dataset.

## 7. Limitations and Risks

- The dataset is historical and may not represent current fraud patterns.
- Approximately 75.58% of transactions lack identity data, limiting the availability of those signals.
- The observed test precision of approximately 74.49% is below the validation target.
- Performance may vary across time periods and transaction subgroups.
- The reported metrics do not establish fairness, calibration, production latency, or real-world business impact.
- The model should not be treated as a fully validated autonomous decision system.

## 8. Reproducibility and Lineage

- **Winning run ID:** `b5bc0037a1b04663adebf8a6fc78643d`
- **Git commit recorded:** `404aca2c9b9b1bee238f362bc33ff02d72a9576a`
- **Git worktree at training time:** Dirty
- **Transaction dataset DVC MD5:** `58b4038d8715f5e11007b826bef00ce7`
- **Identity dataset DVC MD5:** `8487db5001c8bad139f3318d5d3db416`
- **Feature list:** `docs/validated_selected_features.csv`
- **Experiment summary:** `docs/phase3_experiment_summary.json`

The recorded Git worktree was dirty, so the commit alone does not fully identify the exact source state used for training. Preserve the experiment artifacts and record the final committed code state separately.
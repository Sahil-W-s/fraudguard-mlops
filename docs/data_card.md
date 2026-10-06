# FraudGuard — IEEE-CIS Data Card

## 1. Dataset Overview

### Dataset
IEEE-CIS Fraud Detection Dataset

### Source
The dataset was obtained from Kaggle and contains transaction and identity tables.

### Files Used

For Phase 1, only the training datasets are used:

- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

Test datasets are not used during Phase 1.

### Dataset Size

| Dataset | Rows | Columns | Approx. File Size |
|---|---:|---:|---:|
| train_transaction.csv | 590,540 | 394 | ~683 MB |
| train_identity.csv | 144,233 | 41 | ~26.5 MB |
| Joined dataset | 590,540 | 434 | In-memory only |

The joined dataset contains 434 columns because `TransactionID` is shared between the two tables.

---

## 2. Data Join

The transaction and identity datasets are joined using:

`TransactionID`

A **left join** is used so that every transaction remains in the dataset, including transactions for which no corresponding identity information exists.

### Join Results

- Transactions before join: **590,540**
- Transactions after join: **590,540**
- Transactions with identity data: **144,233**
- Transactions without identity data: **446,307**
- Identity-less transactions: **75.58%**

The join is implemented in:

`src/ingestion/load_data.py`

Duplicate `TransactionID` checks are performed before the join.

---

## 3. Feature Groups

### Transaction Features

The transaction table contains:

- Transaction identifier: `TransactionID`
- Fraud label: `isFraud`
- Transaction amount: `TransactionAmt`
- Transaction time information: `TransactionDT`
- Card-related features
- Address-related features
- Distance-related features
- Categorical and anonymous `C*`, `M*`, `D*`, `V*` feature groups

### Identity Features

The identity table contains:

- `TransactionID`
- Device-related features
- Browser-related features
- Operating-system-related features
- Network/device identity information
- Anonymous `id_*` feature groups

Many feature names are anonymized, so their exact real-world meaning is not available directly from the column names.

---

## 4. Label Definition

The target variable is:

`isFraud`

Values:

- `0` — legitimate transaction
- `1` — fraudulent transaction

Observed fraud rate:

**3.4990%**

This is consistent with the expected approximately 3.5% fraud rate for the dataset.

---

## 5. Missing-Value Analysis

A null-rate of **90%** was used as the analysis threshold.

Columns exceeding 90% missing values were investigated rather than automatically removed.

### Transaction Features

| Feature | Missing Rate | Decision |
|---|---:|---|
| `dist2` | 93.63% | Keep for now |
| `D7` | 93.41% | Keep for now |

### Identity Features

| Feature | Missing Rate | Decision |
|---|---:|---|
| `id_24` | 96.71% | Keep for now |
| `id_25` | 96.44% | Keep for now |
| `id_07` | 96.43% | Keep for now |
| `id_08` | 96.43% | Keep for now |
| `id_21` | 96.42% | Keep for now |
| `id_26` | 96.42% | Keep for now |
| `id_23` | 96.42% | Keep for now |
| `id_27` | 96.42% | Keep for now |
| `id_22` | 96.42% | Keep for now |

These features are retained during Phase 1 because missingness/presence shows a measurable association with the fraud label.

For example:

- `D7`: 2.70% fraud when missing vs 14.88% when present
- `dist2`: 3.06% fraud when missing vs 9.92% when present
- `id_24`: 3.46% fraud when missing vs 8.47% when present

These results represent univariate associations and do not by themselves establish predictive or causal importance.

Feature selection and possible removal will be reconsidered during later phases.

---

## 6. Data Validation

The standalone validation script is:

`src/validation/validate_data.py`

The following checks are currently implemented:

### Schema Validation

Checks for required columns including:

- `TransactionID`
- `isFraud`
- `TransactionAmt`
- `TransactionDT`

and the required `TransactionID` column in the identity table.

### Data Type Validation

Critical numerical fields are checked for appropriate numeric types.

### Range Validation

The following checks include:

- `TransactionAmt > 0`
- `isFraud` must contain only `0` or `1`

### Null-Rate Validation

The current pipeline hard-fails if a column exceeds:

**99% missing values**

The 90% threshold is used for analysis and documentation, while the 99% threshold is the current pipeline validation threshold.

### Class-Balance Validation

Expected fraud rate:

**3.5%**

Accepted range:

**2.5% – 4.5%**

Observed:

**3.4990%**

Therefore, the class-balance validation passes.

### Uniqueness Validation

`TransactionID` is verified to be unique in both transaction and identity tables.

---

## 7. Validation Failure Test

The validation pipeline was tested using intentionally corrupted data.

A transaction's `TransactionAmt` was changed from:

`68.5`

to:

`-100.0`

The validator correctly rejected the corrupted dataset during range validation.

This confirms that the validation pipeline fails loudly when invalid transaction values are introduced.

Test file:

`tests/data_validation/test_validation_failure.py`

---

## 8. Data Versioning

DVC is used to version the raw datasets.

The datasets are tracked using:

- `train_transaction.csv.dvc`
- `train_identity.csv.dvc`

The DVC remote is currently configured as local storage and can later be changed to cloud storage.

Current DVC status:

`Data and pipelines are up to date.`

The large CSV files themselves are not committed directly to Git; their DVC pointer files are committed instead.

---

## 9. Known Limitations and Data Considerations

### High Missingness

Several features contain more than 90% missing values. Missingness itself may contain useful fraud-related information, so these features are not automatically discarded.

### Identity Coverage

Approximately **75.58% of transactions do not have corresponding identity records**.

This means models relying heavily on identity information must account for the large identity-less population.

### Class Imbalance

Fraudulent transactions represent approximately **3.5%** of the dataset. Fraud detection models therefore operate under significant class imbalance.

### Anonymized Features

Many features have anonymized names, making their real-world interpretation difficult without additional dataset documentation.

### Benchmark Dataset

The dataset is suitable for developing and evaluating the FraudGuard pipeline, but results obtained from this dataset should not automatically be assumed to represent performance on live production transaction data.

---

## 10. Time Information

The dataset contains the `TransactionDT` field, which represents transaction timing information.

The exact calendar-date range has not yet been derived from `TransactionDT`.

This will be converted into a human-readable time range during a later data-analysis step rather than assuming calendar dates.

---

## 11. Phase 1 Status

Phase 1 objectives completed:

- [x] IEEE-CIS transaction and identity data acquired
- [x] Raw data stored under DVC
- [x] DVC remote configured
- [x] Git pointer files committed
- [x] Transaction/identity join implemented
- [x] Identity-less transaction count documented
- [x] Null analysis completed
- [x] High-null feature decisions documented
- [x] Schema validation implemented
- [x] Range validation implemented
- [x] Null-rate validation implemented
- [x] Class-balance validation implemented
- [x] TransactionID uniqueness validation implemented
- [x] Standalone validation script implemented
- [x] Corrupted-data failure test completed
- [x] Data card documented
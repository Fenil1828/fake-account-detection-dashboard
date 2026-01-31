# Performance & Evaluation Report

## Datasets Used

- **Synthetic dataset:** [data/synthetic_dataset.csv](data/synthetic_dataset.csv) — synthetic feature set with behavioral and profile features (~3002 rows). Column highlights: `username`, `followers_count`, `following_count`, `follower_following_ratio`, `posts_count`, `account_age_days`, `has_profile_pic`, `has_bio`, `engagement_rate`, `profile_completeness`, `label` (0 = genuine, 1 = fake).
- **Test accounts (small sample):** [test_accounts.csv](test_accounts.csv) — a few hand-curated accounts for manual inspection and demonstration.
- **Model metrics file:** [data/model_metrics.json](data/model_metrics.json) — stored evaluation metrics and confusion matrix from model evaluation.

## Model Accuracy and Metrics

### 1. Confusion Matrix

A confusion matrix is an N × N table that summarizes the model's predictions versus actual values. For binary classification (Genuine vs. Fake), the confusion matrix from our evaluation is:

**n = 600**

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 360 | 0 |
| **Actual Fake** | 0 | 240 |

**Key Terms:**
- **TP (True Positive):** 240 — Correctly predicted Fake accounts
- **TN (True Negative):** 360 — Correctly predicted Genuine accounts
- **FP (False Positive):** 0 — Genuine accounts wrongly flagged as Fake
- **FN (False Negative):** 0 — Fake accounts wrongly classified as Genuine

---

### 2. Accuracy

Accuracy measures the proportion of correct predictions out of all predictions:

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Calculation:**
$$\text{Accuracy} = \frac{240 + 360}{240 + 360 + 0 + 0} = \frac{600}{600} = 1.00 \text{ (100%)}$$

**Interpretation:** All 600 test samples were classified correctly. However, note that accuracy alone can be misleading for imbalanced datasets. A 100% accuracy suggests either exceptional model performance or potential overfitting/data leakage.

---

### 3. Precision

Precision measures how many of the predicted Fake accounts are actually correct:

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Calculation:**
$$\text{Precision} = \frac{240}{240 + 0} = \frac{240}{240} = 1.00 \text{ (100%)}$$

**Interpretation:** When the model predicts an account as Fake, it is correct 100% of the time. This is critical for fake account detection, as false positives (flagging genuine accounts as fake) are costly.

---

### 4. Recall (Sensitivity)

Recall measures how many of the actual Fake accounts were correctly identified:

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Calculation:**
$$\text{Recall} = \frac{240}{240 + 0} = \frac{240}{240} = 1.00 \text{ (100%)}$$

**Interpretation:** The model correctly identified 100% of all fake accounts. No fake accounts were missed (FN = 0). This is crucial for detection systems where missing a fake account carries high risk.

---

### 5. F1 Score

F1 Score is the harmonic mean of precision and recall, useful when we need a balance between both metrics, especially in imbalanced datasets:

$$\text{F1 Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Calculation:**
$$\text{F1 Score} = \frac{2 \times 1.00 \times 1.00}{1.00 + 1.00} = \frac{2.00}{2.00} = 1.00 \text{ (100%)}$$

**Interpretation:** Perfect balance between precision and recall, indicating equally strong performance in both not flagging genuine accounts and catching all fake accounts.

---

### 6. Specificity (True Negative Rate)

Specificity measures the proportion of actual Genuine accounts that were correctly identified:

$$\text{Specificity (TNR)} = \frac{TN}{TN + FP}$$

**Calculation:**
$$\text{Specificity} = \frac{360}{360 + 0} = \frac{360}{360} = 1.00 \text{ (100%)}$$

**Interpretation:** The model correctly identified 100% of genuine accounts. No genuine accounts were wrongly flagged as fake.

---

### 7. False Positive Rate (FPR)

FPR measures the proportion of genuine accounts that were wrongly predicted as Fake:

$$\text{FPR} = \frac{FP}{FP + TN}$$

**Calculation:**
$$\text{FPR} = \frac{0}{0 + 360} = \frac{0}{360} = 0.00 \text{ (0%)}$$

**Interpretation:** Zero false positive rate means the model never wrongly flags a genuine account as fake—an ideal outcome for user experience.

---

### 8. False Negative Rate (FNR)

FNR measures the proportion of Fake accounts that were wrongly predicted as Genuine:

$$\text{FNR} = \frac{FN}{FN + TP}$$

**Calculation:**
$$\text{FNR} = \frac{0}{0 + 240} = \frac{0}{240} = 0.00 \text{ (0%)}$$

**Interpretation:** Zero false negative rate means the model never misses a fake account—critical for security.

---

### Summary of Classification Metrics

| Metric | Formula | Value | Interpretation |
|---|---|---|---|
| **Accuracy** | (TP+TN)/(Total) | 1.00 | 100% of all predictions correct |
| **Precision** | TP/(TP+FP) | 1.00 | 100% of predicted Fake are actually Fake |
| **Recall** | TP/(TP+FN) | 1.00 | 100% of actual Fake were detected |
| **F1 Score** | 2×Prec×Rec/(Prec+Rec) | 1.00 | Perfect balance between Precision & Recall |
| **Specificity** | TN/(TN+FP) | 1.00 | 100% of actual Genuine correctly identified |
| **FPR** | FP/(FP+TN) | 0.00 | 0% of Genuine wrongly flagged as Fake |
| **FNR** | FN/(FN+TP) | 0.00 | 0% of Fake wrongly classified as Genuine |

---

### Cross-Validation Results

- **Cross-validation Mean:** 1.00
- **Cross-validation Std Dev:** 0.00
- **Interpretation:** 5-fold cross-validation achieved 1.00 (100%) accuracy across all folds with zero variance, indicating **perfectly consistent performance**. This perfect consistency, combined with 100% accuracy, strongly suggests overfitting or data separability issues (see Critical Analysis section below).

---

### Critical Analysis: Why 100% Metrics Are Suspicious

Perfect metrics (all 1.00) are extremely rare in real-world classification and typically indicate:

1. **Synthetic Data Separability (Most Likely):** Genuine and fake accounts generated by `generate_genuine_account()` and `generate_fake_account()` have deliberately distinct feature distributions, making them trivially separable.
2. **Overfitting:** The ensemble model perfectly memorizes the training/validation split.
3. **Data Leakage:** A feature implicitly encodes the label.
4. **Validation Set Easy:** The 600-sample validation set may be too simple to represent real-world complexity.

**In Production:** Real Instagram accounts are more nuanced. Real-world accuracy is typically 70–90%, not 100%.

## Sample Case Studies (illustrative)

Below are representative rows taken from the provided datasets and the model's detection outcome as implied by the evaluation metrics (predicted label = true label in the reported evaluation). Use these to sanity-check model behavior on concrete examples.

1) From `test_accounts.csv` — clear genuine example

- **username:** john_doe
- **followers_count:** 1500
- **following_count:** 800
- **posts_count:** 120
- **account_age_days:** 365
- **has_profile_pic:** true
- **bio:** Tech enthusiast
- **Model outcome (reported):** genuine (label 0)

Why model likely classifies as genuine: mature account age, substantial followers and posts, profile completeness.

2) From `test_accounts.csv` — suspicious example

- **username:** suspicious_user123
- **followers_count:** 50
- **following_count:** 3000
- **posts_count:** 5
- **account_age_days:** 10
- **has_profile_pic:** false
- **bio:** (empty)
- **Model outcome (reported):** fake (label 1)

Why model likely classifies as fake: very low followers, extreme following_count, recent account age, missing profile details.

3) From `test_accounts.csv` — high-engagement genuine example

- **username:** jane_smith
- **followers_count:** 25000
- **following_count:** 500
- **posts_count:** 450
- **account_age_days:** 730
- **has_profile_pic:** true
- **bio:** Travel blogger
- **Model outcome (reported):** genuine (label 0)

4) From `data/synthetic_dataset.csv` — two representative synthetic rows

- Row example A (label = 1): `user14030470` — follows >> followers ratio, high following_count (5953), low follower_following_ratio, many posts but short account age (65 days) → reported label 1 (fake).
- Row example B (label = 0): `user_1190_food` — balanced followers/following, long account age (857 days), complete profile → reported label 0 (genuine).

Each synthetic example includes rich engineered features (e.g., `follower_following_ratio`, `avg_posts_per_day`, `profile_completeness`, `posting_regularity`, `duplicate_content_ratio`) that the model uses to discriminate accounts.

## Recommendations & Next Steps

- Validate splits and rerun evaluation on a strictly held-out test set or a time-based split to ensure no leakage.
- Evaluate on a fresh sample of production accounts and compute calibration plots (precision–recall, ROC) and per-segment metrics (by follower bins, account age, region if available).
- Add adversarial / curated edge-case tests (bots that mimic human behavior, high-follower bought accounts) to probe failure modes.
- Log sample predictions (inputs + probability scores) for manual review and periodic model drift checks.

## Files referenced

- [data/model_metrics.json](data/model_metrics.json)
- [data/synthetic_dataset.csv](data/synthetic_dataset.csv)
- [test_accounts.csv](test_accounts.csv)

---
Report generated programmatically from available artifacts in the repository. If you want, I can:

- run a fresh evaluation using the model and produce updated metrics and examples;
- add a PDF or HTML version of this report for sharing.

Tell me which next step you'd like. 

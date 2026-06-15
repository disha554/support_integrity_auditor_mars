# Support Integrity Auditor (SIA)

## Project Overview

Support Integrity Auditor (SIA) is an automated auditing system designed to identify inconsistencies between the **human-assigned priority** and the **actual severity** of customer support tickets.

The system infers ticket severity using textual and metadata signals and flags tickets where the assigned priority appears inconsistent with the inferred severity.

---

## Key Features

Automated Priority Mismatch Detection

Self-Supervised Pseudo Label Generation

NLP-Based Ticket Analysis

Metadata-Aware Classification

Evidence Dossier Generation

Streamlit Web Application

---

## Dataset

**Customer Support Tickets CRM Dataset**

### Dataset Statistics

| Metric | Value |
|----------|---------|
| Total Records | 20,000 |
| Features | 12 |
| Problem Type | Binary Classification |
| Labels | Pseudo-Generated |

### Features Used

| Feature | Description |
|----------|------------|
| Ticket_Subject | Short summary of issue |
| Ticket_Description | Detailed issue description |
| Priority_Level | Human-assigned priority |
| Ticket_Channel | Ticket source |
| Issue_Category | Category of issue |
| Resolution_Time_Hours | Time required for resolution |
| Satisfaction_Score | Customer satisfaction rating |

---

## Project Pipeline

```text
Raw Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Pseudo Label Generation
     │
     ▼
TF-IDF + Metadata Features
     │
     ▼
XGBoost Classifier
     │
     ▼
Priority Mismatch Detection
     │
     ▼
Evidence Dossier Generation
```

---

## Stage 1: Data Preprocessing

### Text Processing

- Lowercasing
- Stopword Removal
- Lemmatization
- Punctuation Removal
- Text Normalization

### Feature Engineering

Created the following features:

- Combined Ticket Text
- Word Count
- Character Count
- Resolution Severity
- Satisfaction Severity
- Keyword Severity

---

## Stage 2: Pseudo Label Generation

Since no ground-truth mismatch labels were available, pseudo-labels were generated using multiple severity signals.

### Signal 1: Resolution-Time Severity

| Resolution Time | Severity |
|---------------|-----------|
| ≤ 12 hrs | Low |
| 12–24 hrs | Medium |
| 24–48 hrs | High |
| > 48 hrs | Critical |

### Signal 2: Keyword Severity

Critical Keywords:

```text
outage
server down
payment failed
security breach
account locked
```

High Severity Keywords:

```text
crash
error
failure
unable
urgent
```

### Signal 3: Satisfaction Severity

Lower satisfaction scores indicate higher issue severity.

### Severity Fusion

```python
severity_score = (
    0.4 * resolution_signal +
    0.4 * keyword_signal +
    0.2 * satisfaction_signal
)
```

---

## Stage 3: Model Training

### Text Features

TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    min_df=2
)
```

### Metadata Features

- Resolution_Time_Hours
- Satisfaction_Score
- Ticket_Channel
- Issue_Category
- Word_Count

### Classifier

```python
XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05
)
```

### Imbalance Handling

- Class Weighting
- Threshold Tuning
- Recall Optimization

---

## Model Performance

### Final Results

| Metric | Score |
|----------|--------|
| Accuracy | **78.98%** |
| Macro F1 Score | **0.75** |
| Recall (Class 0) | **0.80** |
| Recall (Class 1) | **0.77** |

### Confusion Matrix

```text
[[2397 614]
 [ 227 762]]
```

---

## Evidence Dossier Generation

Each detected mismatch generates a structured evidence dossier.

### Example Schema

```json
{
  "ticket_id": "...",
  "assigned_priority": "...",
  "inferred_severity": "...",
  "mismatch_type": "...",
  "severity_delta": "...",
  "feature_evidence": [],
  "constraint_analysis": "...",
  "confidence": "..."
}
```

---

## Streamlit Application

Run locally:

```bash
streamlit run app.py
```

### Features

- Single Ticket Analysis
- Mismatch Detection
- Confidence Score
- Evidence Dossier Generation

---

## Repository Structure

```text
Support_Integrity_Auditor/

├── EDA.ipynb
├── Data_Preprocessing.ipynb
├── Pseudo_Label_Generation.ipynb
├── Model_Training_XGBoost.ipynb

├── pseudo_labeled_dataset.csv

├── sia_xgboost.pkl
├── tfidf.pkl

├── evidence_dossiers.json

├── app.py
├── requirements.txt

└── README.md
```

---

## Installation

```bash
git clone <repository-url>

cd Support_Integrity_Auditor

pip install -r requirements.txt
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- NLTK
- Streamlit
- Matplotlib
- Seaborn

---

## Future Improvements

- Fine-Tuned DeBERTa-v3-small
- LoRA-Based Adaptation
- LLM-Powered Evidence Generation
- Adversarial Ticket Testing
- Real-Time Enterprise Deployment

---

## Author

Disha Agarwal  
Chemical Engineering, IIT Roorkee

---


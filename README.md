# Credit Risk Machine Learning System

An end-to-end, production-ready Credit Risk Assessment System that spans the entire machine learning lifecycle—from exploratory data analysis and handling severe class imbalances to probability calibration, model explainability, and a live web interface dashboard.

## Project Architecture & Lifecycle

### 1. Data Preparation & EDA
* **Ingestion:** Loaded credit risk datasets directly from Hugging Face.
* **Data Sanitization:** Handled extreme anomalies and filtered invalid ranges (e.g., locking age thresholds between 18 and 100, cleansing unrealistic employment lengths and out-of-bounds loan metrics).
* **Pipeline Construction:** Built distinct automated preprocessing pipelines using Scikit-Learn's ColumnTransformer—applying median imputation for numerical data and tracking categorical variations with One-Hot Encoding.

### 2. Model Training & Class Imbalance Mitigation
The dataset features a heavy class imbalance (far fewer default cases than non-defaults). To address this, a comparative analysis was performed across two distinct architectures:
* **Logistic Regression:** Regularized linear baseline using `class_weight='balanced'` to offset minority class penalties.
* **XGBoost Classifier:** Advanced gradient-boosted decision tree architecture utilizing targeted fractional adjustments for `scale_pos_weight` based on native balance calculations.
* **Cross-Validation:** Multi-fold validation loops used across all iterations to prevent out-of-sample data leakage and overfitting.

### 3. Optimization & Performance Tuning
* **Hyperparameter Search:** Used `RandomizedSearchCV` to optimize XGBoost spaces across continuous distributions for learning rates, max tree depth, subsample ratios, and structural regularization constraints like `gamma`.
* **Threshold Adjustment:** Instead of defaulting to standard 0.5 decision boundaries, precision-recall space curves were analyzed to optimize the threshold configuration specifically around maximizing the F1-Score.

### 4. Calibration & Interpretability
* **Probability Calibration:** Boosted models like XGBoost can be overly confident in their output scores. Platt Scaling (Sigmoid calibration via `CalibratedClassifierCV`) was implemented to align prediction percentages directly with real-world probability vectors.
* **SHAP Explainability:** Integrated SHAP (SHapley Additive exPlanations) to turn the black-box ensemble model transparent. Summary plots illustrate global feature impact, while interactive Waterfall plots map precisely why an individual applicant was flagged or cleared.

### 5. API Development & Frontend UI
* **FastAPI Backend:** Built a high-performance web service wrapped in a clean async lifespan architecture to load serialized artifacts safely.
* **Data Validation:** Utilized Pydantic schemas to strictly enforce type constraints on payload submissions (e.g., age, income, and debt-to-income limits).
* **Static Dashboard:** Set up a local HTML/CSS/JS template interface mounted directly onto the root route to let users submit applicant details and instantly display risk profiles back from the engine.

---

## Technical Stack
* **Language:** Python
* **Modeling & Infrastructure:** Scikit-Learn, XGBoost, Joblib, Pandas, NumPy
* **Explainability:** SHAP
* **API Framework:** FastAPI, Uvicorn, Pydantic
* **Deployment Config:** Render specification (`render.yaml`), requirements tracking

---

## Production Deployment & Hosting

This project is fully configured for automated cloud environments using the included **`render.yaml`** blueprint. When connected to a web service platform like Render or Railway, the infrastructure reads the configuration file to instantly provision a Python container environment, handle dependency mapping via `requirements.txt`, and spin up the live production web app.

### API Reference & Testing
Once deployed live, you can interact with the system via the interactive documentation dashboard at:
* `https://riskshield-xgboost.onrender.com`

#### Sample JSON Input Payload (`POST /predict`)
```json
{
  "person_age": 24,
  "person_income": 58000,
  "person_home_ownership": "RENT",
  "person_emp_length": 2.0,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.2,
  "loan_percent_income": 0.17
}
```



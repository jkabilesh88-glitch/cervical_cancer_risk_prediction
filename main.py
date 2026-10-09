# ============================================================
# CERVICAL CANCER RISK PREDICTION USING RANDOM FOREST
# ============================================================

# STEP 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# STEP 2: LOAD DATASET
# ------------------------------------------------------------

print("\nSTEP 1: Loading Dataset")

df = pd.read_csv("cervical_cancer.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# STEP 3: DATA PREPROCESSING
# ------------------------------------------------------------

print("\nSTEP 2: Data Preprocessing")

# Replace missing-value symbols with NaN
df.replace(["?", " "], np.nan, inplace=True)

# Set the target column
TARGET_COLUMN = "Biopsy"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target '{TARGET_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Convert target to numeric and remove rows with missing target
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN], errors="coerce"
)
df.dropna(subset=[TARGET_COLUMN], inplace=True)

# Separate input features (X) and target (y)
X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].astype(int)

# Convert features to numeric values
for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Remove columns that contain no usable values
X.dropna(axis=1, how="all", inplace=True)

# Check for binary classification
if y.nunique() != 2 or not set(y.unique()).issubset({0, 1}):
    raise ValueError(
        "This program expects a binary target encoded as 0 and 1."
    )

# Display target distribution
print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nInput features:", X.shape[1])
print("Missing feature values:", int(X.isna().sum().sum()))

# ------------------------------------------------------------
# STEP 4: CLASSIFICATION METHODOLOGY
# Apply Random Forest Classification
# ------------------------------------------------------------

print("\nSTEP 3: Random Forest Classification")

# Random Forest will be our classification algorithm
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

print("Random Forest classifier created.")

# ------------------------------------------------------------
# STEP 5: FEATURE SCALING
# Random Forest does not require feature scaling
# ------------------------------------------------------------

print("\nSTEP 4: Feature Scaling")

print("Feature scaling is NOT required for Random Forest.")
print("Original feature values will be used.")

# ------------------------------------------------------------
# STEP 6: SPLIT DATA AND TRAIN BASELINE MODEL
# ------------------------------------------------------------

print("\nSTEP 5: Model Training")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Impute missing values using training data only
baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", rf_model)
])

# Train the baseline model
baseline_model.fit(X_train, y_train)

# Predict test results before tuning
y_pred_before = baseline_model.predict(X_test)

print("Baseline model trained successfully!")

# ------------------------------------------------------------
# STEP 7: HYPERPARAMETER TUNING USING GRIDSEARCHCV
# ------------------------------------------------------------

print("\nSTEP 6: Hyperparameter Tuning")

# Parameters to test
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

# Five-fold cross-validation
grid_search = GridSearchCV(
    estimator=baseline_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best model selected by GridSearchCV
best_model = grid_search.best_estimator_

# Predict test results after tuning
y_pred_after = best_model.predict(X_test)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:")
print(round(grid_search.best_score_, 4))

# ------------------------------------------------------------
# STEP 8: PERFORMANCE EVALUATION
# Compare before and after tuning
# ------------------------------------------------------------

print("\nSTEP 7: Performance Evaluation")


def evaluate_model(y_true, y_pred, model_name):
    """Calculate classification performance metrics."""

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }

    return metrics


before_metrics = evaluate_model(
    y_test, y_pred_before, "Before Tuning"
)

after_metrics = evaluate_model(
    y_test, y_pred_after, "After Tuning"
)

# Display comparison table
results = pd.DataFrame([before_metrics, after_metrics])

print("\nPerformance Comparison:")
print(results.to_string(index=False, float_format="%.4f"))

print("\nClassification Report After Tuning:")
print(classification_report(
    y_test,
    y_pred_after,
    zero_division=0
))

# ------------------------------------------------------------
# STEP 9: VISUALIZATION
# Confusion Matrix and Feature Importance
# ------------------------------------------------------------

print("\nSTEP 8: Visualization")

# Plot confusion matrices before and after tuning
cm_before = confusion_matrix(
    y_test, y_pred_before, labels=[0, 1]
)

cm_after = confusion_matrix(
    y_test, y_pred_after, labels=[0, 1]
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    cm_before,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Risk", "Risk"],
    yticklabels=["No Risk", "Risk"],
    ax=axes[0]
)
axes[0].set_title("Confusion Matrix - Before Tuning")
axes[0].set_xlabel("Predicted Class")
axes[0].set_ylabel("Actual Class")

sns.heatmap(
    cm_after,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Risk", "Risk"],
    yticklabels=["No Risk", "Risk"],
    ax=axes[1]
)
axes[1].set_title("Confusion Matrix - After Tuning")
axes[1].set_xlabel("Predicted Class")
axes[1].set_ylabel("Actual Class")

plt.tight_layout()
plt.show()

# Feature importance from the tuned Random Forest
rf = best_model.named_steps["classifier"]

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot the top 10 important features
top_features = feature_importance.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 10: FINAL OUTPUT AND CLASSIFICATION RESULTS
# ------------------------------------------------------------

print("\nSTEP 9: Final Classification Results")

# Show actual vs predicted results for the test dataset
final_results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred_after,
    "Prediction": [
        "Risk Class (1)" if p == 1 else "No Risk Class (0)"
        for p in y_pred_after
    ]
})

print("\nFirst 20 Test Predictions:")
print(final_results.head(20).to_string(index=False))

# Display best-performing model's test metrics
print("\nFINAL MODEL PERFORMANCE")
print(f"Accuracy : {after_metrics['Accuracy']:.4f}")
print(f"Precision: {after_metrics['Precision']:.4f}")
print(f"Recall   : {after_metrics['Recall']:.4f}")
print(f"F1-Score : {after_metrics['F1-Score']:.4f}")

print("\nProgram completed successfully!")
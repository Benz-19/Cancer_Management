import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.data_processor import load_data
from utils.env_error import reset

from agents.feature_selection_agent import FeatureSelectionAgent
from agents.classifier_agent import ClassifierAgent
from agents.outlier_agent import OutlierAgent
from agents.expert_agent import ExpertAgent
from agents.drug_supply_agent import DrugSupplyAgent
from agents.treatment_clustering_agent import TreatmentClusteringAgent
from agents.evaluation_agent import EvaluationAgent

reset()
data = load_data("data/Cancer_Data.csv")

X = data.drop(["id", "diagnosis"], axis=1)
y = LabelEncoder().fit_transform(data["diagnosis"])

print(f"\nTotal Patients in System: {len(X)}")

# ================= FEATURE SELECTION =================
print("\nSelecting Features")
fs_agent = FeatureSelectionAgent()
features = fs_agent.wrapper_method(X, y)

X_selected = X[features]

# ================= CLASSIFICATION =================
print("\nTraining Classifier")
clf_agent = ClassifierAgent()
model, X_test, y_test = clf_agent.train(X_selected, y)

predictions = clf_agent.predict(X_selected)

# ================= OUTLIERS =================
reset()
print("\nFinding Outliers")
outlier_agent = OutlierAgent()
outliers = outlier_agent.find_outliers(X_selected.values)

# ================= EXPERT SYSTEM =================
reset()
print("\nExpert System Evaluating")
expert = ExpertAgent()
critical_patients = expert.critical_cases(predictions, outliers)

print(f"Total Critical Patients: {len(critical_patients)}")

X_critical = X.iloc[critical_patients]
y_critical = y[critical_patients]

# ================= CRITICAL FEATURE SELECTION =================
critical_features = fs_agent.wrapper_method(
    X_critical, y_critical, max_features=5
)

print("\nCritical-only Features:", critical_features)

# ================= TREATMENT CLUSTERING =================
treatment_agent = TreatmentClusteringAgent()
labels = treatment_agent.cluster(X_critical[critical_features])

drug_patients = []
advice_patients = []

for i, label in enumerate(labels):
    pid = critical_patients[i]
    if label == 1:
        drug_patients.append(pid)
    else:
        advice_patients.append(pid)

print("\nPatients Given Drug:", drug_patients)
print("Patients Given Advice:", advice_patients)

# ================= DRUG SUPPLY =================
drug_agent = DrugSupplyAgent()
drug_agent.treat(drug_patients)

# ================= EVALUATION =================
eval_agent = EvaluationAgent()
eval_agent.show_confusion_matrix(
    model, X_test, y_test, "Random Forest Confusion Matrix"
)

# ================= SUMMARY =================
print("\nSUMMARY")
print(f"Total Patients: {len(X)}")
print(f"Selected Features: {features}")
print(f"Critical Patients: {len(critical_patients)}")
print(f"Critical-only Features: {critical_features}")

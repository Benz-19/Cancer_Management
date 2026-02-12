import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.data_processor import load_data
from utils.env_error import reset
import time

from agents.feature_selection_agent import FeatureSelectionAgent
from agents.classifier_agent import ClassifierAgent
from agents.outlier_agent import OutlierAgent
from agents.expert_agent import ExpertAgent
from agents.drug_supply_agent import DrugSupplyAgent
from agents.treatment_clustering_agent import TreatmentClusteringAgent
from agents.evaluation_agent import EvaluationAgent
from agents.decision_rule_agent import DecisionRuleAgent

# SYSTEM START
reset()
system_start = time.time()

data = load_data("data/Cancer_Data.csv")

X = data.drop(["id", "diagnosis"], axis=1)
y = LabelEncoder().fit_transform(data["diagnosis"])

print(f"\nTotal Patients in System: {len(X)}")

# FEATURE SELECTION
print("\nSelecting Features")
fs_agent = FeatureSelectionAgent()
features = fs_agent.wrapper_method(X, y)

X_selected = X[features]

# CLASSIFICATION
print("\nTraining Classifier")
clf_agent = ClassifierAgent()
model, X_test, y_test = clf_agent.train(X_selected, y)

predictions = clf_agent.predict(X_selected)

# OUTLIER DETECTION
reset()
print("\nFinding Outliers")
outlier_agent = OutlierAgent()
outliers = outlier_agent.find_outliers(X_selected.values)

# EXPERT SYSTEM (CRITICAL CASES)
reset()
print("\nExpert System Evaluating")
expert = ExpertAgent()
critical_patients = expert.critical_cases(predictions, outliers)

print(f"Total Critical Patients: {len(critical_patients)}")

X_critical = X.iloc[critical_patients]
y_critical = y[critical_patients]

# CRITICAL FEATURE SELECTION
critical_features = fs_agent.wrapper_method(
X_critical, y_critical, max_features=5
)

print("\nCritical-only Features:", critical_features)

# SAVE CRITICAL DATA
critical_global_df = X.loc[critical_patients, features].copy()
critical_global_df["Patient_Index"] = critical_patients
critical_global_df.to_csv(
    "data/results/critical_patients_global_features.csv",
    index=False
)

critical_specific_df = X.loc[critical_patients, critical_features].copy()
critical_specific_df["Patient_Index"] = critical_patients
critical_specific_df.to_csv(
    "data/results/critical_patients_critical_features.csv",
    index=False
)

print("\nSaved critical patient CSV files")

# TREATMENT CLUSTERING
print("\nClustering Critical Patients for Treatment Decision")
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

#   DECISION RULE SYSTEM  
print("\nTraining Decision Rule Expert System")

rule_agent = DecisionRuleAgent()
rule_agent.train(
    X_critical[critical_features],
    labels
)

# DRUG SUPPLY
print("\nAdministering Drugs")
drug_agent = DrugSupplyAgent()
drug_agent.treat(drug_patients)

# EVALUATION
eval_agent = EvaluationAgent()
eval_agent.show_confusion_matrix(
    model, X_test, y_test, "Random Forest Confusion Matrix"
)

# SUMMARY
system_end = time.time()

print("\nSUMMARY")
print(f"Total Patients: {len(X)}")
print(f"Selected Features: {features}")
print(f"Critical Patients: {len(critical_patients)}")
print(f"Critical-only Features: {critical_features}")
print(f"Total System Execution Time: {system_end - system_start:.2f} seconds")

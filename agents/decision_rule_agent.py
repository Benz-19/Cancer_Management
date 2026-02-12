from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
import time

class DecisionRuleAgent:
    def __init__(self):
        # Cost-sensitive decision tree
        self.model = DecisionTreeClassifier(
            max_depth=3,
            class_weight={0: 1, 1: 3},  # 1 = Drug (higher cost)
            random_state=42
        )

    def train(self, X, y):
        start = time.time()

        self.model.fit(X, y)
        preds = self.model.predict(X)

        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)

        print("\nDecision Rule Expert System")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print(f"Decision Rule Training Time: {time.time() - start:.2f} seconds")

        # DISPLAY RULES
        rules = export_text(self.model, feature_names=list(X.columns))
        print("\nLearned IF–THEN Rules:\n")
        print(rules)

        # SAVE RULES
        with open("data/results/decision_rules.txt", "w") as f:
            f.write(rules)

        print("\nRules saved to data/results/decision_rules.txt")

        return self.model

    def predict(self, X):
        return self.model.predict(X)

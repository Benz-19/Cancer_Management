import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ClassifierAgent:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X, y):
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"Classifier Accuracy: {acc:.4f}")

        end_time = time.time()
        print(f"Classification Time: {end_time - start_time:.2f} seconds")

        return self.model,X_test, y_test

    def predict(self, X):
        return self.model.predict(X)

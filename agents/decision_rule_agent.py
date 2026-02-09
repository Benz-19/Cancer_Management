from sklearn.tree import DecisionTreeClassifier

class DecisionRuleAgent:
    def train(self, X, y):
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        return model

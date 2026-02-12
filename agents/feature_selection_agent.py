import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class FeatureSelectionAgent:
    def wrapper_method(self, X, y, max_features=10):
        start_time = time.time()

        clf = RandomForestClassifier(random_state=42)
        selected = []
        remaining = list(range(X.shape[1]))
        best_score = 0

        feature_counts = []
        accuracies = []

        print("Forward Selection Steps:")

        while len(selected) < max_features and remaining:
            scores = []

            for feat in remaining:
                test = selected + [feat]
                score = cross_val_score(
                    clf, X.iloc[:, test], y, cv=5
                ).mean()
                scores.append((feat, score))

            feat, score = max(scores, key=lambda x: x[1])

            print(f"Tested {X.columns[feat]} → Accuracy: {score:.4f}")

            if score > best_score:
                selected.append(feat)
                remaining.remove(feat)
                best_score = score

                feature_counts.append(len(selected))
                accuracies.append(score)

                print(f"Selected {X.columns[feat]}")
            else:
                break

        # Graph
        plt.plot(feature_counts, accuracies, marker="o")
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.title("Wrapper Feature Selection Performance")
        plt.show()

        end_time = time.time()
        print(f"Feature Selection Time: {end_time - start_time:.2f} seconds")

        return X.columns[selected].tolist()

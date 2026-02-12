import time
from sklearn.cluster import KMeans

class TreatmentClusteringAgent:
    def cluster(self, X):
        start_time = time.time()

        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)

        end_time = time.time()
        print(f"Treatment Clustering Time: {end_time - start_time:.2f} seconds")

        return labels

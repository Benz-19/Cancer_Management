from sklearn.cluster import KMeans

class TreatmentClusteringAgent:
    def cluster(self, X):
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)
        return labels

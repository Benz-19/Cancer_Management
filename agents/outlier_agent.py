from sklearn.cluster import KMeans
from utils.env_error import reset
import numpy as np

class OutlierAgent:
    def find_outliers(self, X):
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)

        reset()
        centers = kmeans.cluster_centers_
        distances = np.linalg.norm(X - centers[labels], axis=1)

        threshold = distances.mean() + distances.std()
        outliers = distances > threshold

        return outliers

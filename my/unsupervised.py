# ================== UNSUPERVISED (K-Means, GMM, Hierarchical, DBSCAN) ==================

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import make_blobs #datasets , not dataset, vlovabe dekho
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score

#Create dataset
X, y = make_blobs( #valovabe dekho
    n_samples = 300,
    centers = 3,
    cluster_std = 1.0,
    random_state = 42
)

#Scale 
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#figure 1 : plot original data
plt.figure()
plt.scatter(
    X[:, 0],
    X[:, 1],
    c = y 
)
plt.title("Original data distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

#figure 2 : elbow method
inertia = []

k_range = range(1,10)
for k in k_range:
    kmeans = KMeans(
        n_clusters = k,
        random_state = 42
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure()
plt.plot(
    k_range, #valovabe dekho
    inertia,
    marker = "o"
)    
plt.title("Elbow graph")
plt.xlabel("Number of clusters(K)")
plt.ylabel("Inertia")

#Models
models = [
    ("K-Means", KMeans(n_clusters = 3, random_state= 42)),
    ("GMM", GaussianMixture(n_components = 3, random_state = 42)),
    ("Hieararchical", AgglomerativeClustering(n_clusters = 3)),
    ("DBSCAN", DBSCAN(eps = 0.3, min_samples = 5))
]

model_names = []
scores = []

for name, model in models:
    labels = model.fit_predict(X_scaled) #valovabe dekho

    if len(set(labels)) > 1:
        score = silhouette_score(
            X_scaled,
            labels
        )
    else:
        score = -1
    print("====", name, "====")
    print("Silhouette score : \n", score)

    model_names.append(name)
    scores.append(score)

    plt.figure()
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c = labels
    )        
    plt.title(name + " Clustering ")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

plt.figure()
plt.bar(
    model_names,
    scores
)
plt.title("Comparision graph")
plt.xlabel("Algorithms")
plt.ylabel("Silhouette score")

plt.xticks(rotation = 45)
plt.ylim(-1, 1)

plt.tight_layout()
plt.show()

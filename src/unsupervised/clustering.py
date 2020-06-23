from sklearn import datasets, metrics, cluster, mixture
import time, warnings
import numpy as np


# we apply k-means to the training set and vary the number of clusters
# we compute the inertia, silhouette, etc for each k
def kmeans(X):
    print("2.2.1 K-Means")

    kMeansModels, modelsInertia, modelsSilhouette, modelsCalinski, modelsBouldin = [], [], [], [], []
    for i in range(2, 15):
        model = cluster.KMeans(n_clusters=i, random_state=1).fit(X)
        kMeansModels.append(model)
        modelsInertia.append(model.inertia_)
        modelsSilhouette.append(metrics.silhouette_score(X, model.labels_))
        modelsCalinski.append(metrics.calinski_harabasz_score(X, model.labels_))
        modelsBouldin.append(metrics.davies_bouldin_score(X, model.labels_))

    print("a) Inertia for 2 <= k <= 14 :")
    for i in range(0, 13, 2):
        print("k =", i + 2, ":", modelsInertia[i], end=" | ")
    print()

    print("b) Silhouette for 2 <= k <= 14 :")
    for i in range(0, 13, 2):
        print("k =", i + 2, ":", modelsSilhouette[i], end=" | ")
    print()

    print("c) Calinski Harabaz for 2 <= k <= 14 :")
    for i in range(0, 13, 2):
        print("k =", i + 2, ":", modelsCalinski[i], end=" | ")
    print()

    print("d) Davies Bouldin for 2 <= k <= 14 :")
    for i in range(0, 13, 2):
        print("k =", i + 2, ":", modelsBouldin[i], end=" | ")
    print()


# we apply agglomerative clustering with a fixed k = 3 and different linkage criteria
def agglomerative(trnX, trnY):
    print("2.2.2 Agglomerative Clustering")
    algorithms = {}
    predictions = {}
    efficiency = {}

    # 1 - Parameterize clustering algorithms
    algorithms['Ward Linkage'] = cluster.AgglomerativeClustering(
        n_clusters=3, linkage='ward')
    algorithms['Complete Linkage'] = cluster.AgglomerativeClustering(
        n_clusters=3, linkage='complete')
    algorithms['Average Linkage'] = cluster.AgglomerativeClustering(
        n_clusters=3, linkage='average')
    algorithms['Single Linkage'] = cluster.AgglomerativeClustering(
        n_clusters=3, linkage='single')

    # 2 - Run clustering algorithm and store predictions
    for name in algorithms:
        clustering = algorithms[name]
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            clustering.fit(trnX)
            # measure efficiency
        efficiency[name] = time.time() - t0
        if hasattr(clustering, 'labels_'):
            predictions[name] = clustering.labels_.astype(np.int)
        else:
            predictions[name] = clustering.predict(trnX)

    print("a) Efficiency:")
    for t in efficiency:
        print(t, ":", efficiency[t], "s")

    # metrics for each algorithm
    adjRandScore, adjMutInfScore, mutInfScore, normMutInfoScore, homScore, comScore, meaScore = {}, {}, {}, {}, {}, {}, {}
    for p in predictions:
        adjRandScore[p] = metrics.adjusted_rand_score(trnY, predictions[p])
        adjMutInfScore[p] = metrics.adjusted_mutual_info_score(trnY, predictions[p], average_method='arithmetic')
        mutInfScore[p] = metrics.mutual_info_score(trnY, predictions[p])
        normMutInfoScore[p] = metrics.normalized_mutual_info_score(trnY, predictions[p], average_method='arithmetic')
        homScore[p] = metrics.homogeneity_score(trnY, predictions[p])
        comScore[p] = metrics.completeness_score(trnY, predictions[p])
        meaScore[p] = metrics.v_measure_score(trnY, predictions[p])

    scores = [adjRandScore, adjMutInfScore, mutInfScore, normMutInfoScore, homScore, comScore, meaScore]
    labels = ["Adjusted Random Score", "Adjusted Mutual Info Score", "Mutual Info Score",
              "Normalized Mutual Info Score", "Homogeneity Score", "Completeness Score",
              "Measure Score"]
    print("b) Metrics")
    # we select the best and worst for each metric
    for i in range(len(scores)):
        maxScore = max(scores[i].items(), key=lambda x: x[1])
        minScore = min(scores[i].items(), key=lambda x: x[1])

        print(labels[i])
        print("Best:", maxScore[0], maxScore[1])
        print("Worst:", minScore[0], minScore[1])

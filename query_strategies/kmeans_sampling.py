import numpy as np
from .strategy_rebuild import Strategy
from sklearn.cluster import KMeans


class KMeansSampling(Strategy):
    def __init__(self, dataset, net):
        super(KMeansSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        cluster_learner = KMeans(n_clusters=75)
        cluster_learner.fit(embeddings)
        
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)

        q_idxs = []
        for i in range(75):
            num = int(2000 * np.array(np.arange(embeddings.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argsort()[:]]).flatten().shape[0] /
                      embeddings.shape[0])
            if num == 0:
                num = 1
                q_idxs.append(np.arange(embeddings.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argsort()[:num][0]])
            elif num:
                for j in range(num):
                    q_idxs.append(
                        np.arange(embeddings.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argsort()[:num]].flatten()[j])

        return unlabeled_idxs[np.array(q_idxs).flatten()]

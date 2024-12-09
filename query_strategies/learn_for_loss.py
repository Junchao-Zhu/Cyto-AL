import numpy as np
from .strategy_rebuild import Strategy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from scipy.spatial.distance import pdist
from collections import Counter
from joblib import load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from collections import Counter
from scipy.stats import rankdata
import random
import math

import torch.nn.functional as F


class learn_for_loss(Strategy):
    def __init__(self, data, net):
        super(learn_for_loss, self).__init__(data, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        pred_rank = self.predict_loss(unlabeled_data).numpy().squeeze(1)
        q_idxs = []
        sorted_indices = pred_rank.argsort()[::-1]
        for i in range(205):
            q_idxs.append(sorted_indices[i])


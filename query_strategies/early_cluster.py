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


def generate_list(a, b, n):
    nums = list(range(a, b + 1))
    weights = [1.0 / num for num in nums]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    result = np.random.choice(nums, n, p=normalized_weights)
    return sorted(result.tolist())


class density_cluster(Strategy):
    def __init__(self, data, net):
        super(density_cluster, self).__init__(data, net)
        # exposed hyperparameters
        # self.k_cluster = 50
        self.k_cluster = 50
        # self.density_threshold = 0.075
        self.density_threshold = 0.075
        self.num_ranked_samples = 2050 # default original was 2050

    def get_hparams(self):
        return (self.k_cluster, self.density_threshold, self.num_ranked_samples)
    
    def query(self, n):
        """
        preparation for clustering, including get density, feature and uncertainty
        """
        # get cell density
        # Num of k-means clusters
        #n_cluster = 50
        n_cluster = self.k_cluster
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # get latent feature
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        pca = PCA(n_components=30)
        embeddings_pca = torch.from_numpy(pca.fit_transform(embeddings))

        density = self.get_density(unlabeled_data)

        # get entropy uncertainty
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs + 1e-8)
        uncertainties = (probs * log_probs).sum(1)
        uncertainties = uncertainties.unsqueeze(1)

        # get image uncertainty
        undo_idx, undo_data = self.dataset.get_enhance_data()
        img_uncertainty = self.get_img_uncertainty(undo_data)

        # combined variations
        combined_var = torch.cat([density * 15, embeddings_pca], dim=1)
        combined_var = torch.cat([combined_var, img_uncertainty], dim=1)

        # do k-means cluster
        cluster_learner = KMeans(n_clusters=n_cluster)
        cluster_learner.fit(combined_var)
        cluster_idxs = cluster_learner.predict(combined_var)

        # even_sample
        img_uncertainty = img_uncertainty.squeeze(1)

        q_idxs = []

        density_prob = -rankdata(density.squeeze(1)) / density.squeeze(1).shape[0]

        """
        low_density_filtering: filter out patches whose cellularity is smaller than FIXED_THR!!
        """
        # Tunables (no args, change here if needed)
        FIXED_THR = self.density_threshold            # original low-density threshold
        PCT_FALLBACK = 20                             # fallback percentile if empty (use lowest 20% to be filtered out if fixed threshold yields none)
        TOP_RANKED_SAMPLES = self.num_ranked_samples  # batch size of ranking stage (top-ranked patches acquired when Ranking Learning Module is used)
        
        LOW_QUOTA = 20    # how many low-density samples to add back (exploration)
# ---------------- low-density handling (threshold -> percentile -> lowest-N) ----------------
        # low_density_idxs = np.where(density < 0.075)[0]
        low_density_idxs = np.where(density < FIXED_THR)[0]
        if low_density_idxs.size == 0:
            thr = np.percentile(density, PCT_FALLBACK)
            low_density_idxs = np.where(density <= thr)[0]
        if low_density_idxs.size == 0:
            order = np.argsort(density)  # ascending: lowest first
            low_density_idxs = order[:min(LOW_QUOTA, order.size)]

        print(f"unlabeled size: {len(unlabeled_idxs)}")
        print(f"low_density count: {len(low_density_idxs)} "
            f"({len(low_density_idxs)/len(unlabeled_idxs):.3%})")
        print("density stats:",
            f"min={density.min():.4f}  p5={np.percentile(density,5):.4f}  "
            f"median={np.median(density):.4f}  p95={np.percentile(density,95):.4f}  "
            f"max={density.max():.4f}")
        print("low_density_idxs count: ", int(low_density_idxs.size))
                
        for i in range(n_cluster):
            tmp_cluster = np.arange(embeddings.shape[0])[cluster_idxs == i]
            tmp_idx_ranking = torch.from_numpy(density_prob)[cluster_idxs == i].argsort()
            num = int(TOP_RANKED_SAMPLES * tmp_cluster.flatten().shape[0] / embeddings.shape[0])
            if num != 0 and num <= tmp_cluster.flatten().shape[0]:
                for j in range(num):
                    q_idxs.append(tmp_cluster.flatten()[tmp_idx_ranking[j]])
            elif num > tmp_cluster.flatten().shape[0]:
                for j in range(tmp_cluster.flatten().shape[0]):
                    q_idxs.append(tmp_cluster.flatten()[tmp_idx_ranking[j]])

        q_idxs = np.array(q_idxs).flatten()
        # Everything with density < threshold goes into low_density_idxs, and those indices are removed from the primary (high-cellularity/proportional) picks
        # np.setdiff1d(A, B) returns the sorted unique values in A that are not in B
        q_idxs = np.setdiff1d(q_idxs, low_density_idxs)

        q_low_den_idxs = []
        low_selected = np.random.choice(low_density_idxs, size=20, replace=False)
        for i in range(low_selected.shape[0]):
            q_low_den_idxs.append(low_selected[i])

        q_idxs = np.concatenate((q_idxs, np.array(q_low_den_idxs)), axis=0)

        q_random_idxs = []
        if q_idxs.shape[0] < TOP_RANKED_SAMPLES:
            diff = TOP_RANKED_SAMPLES - q_idxs.shape[0]
            wait_list = np.setdiff1d(np.arange(embeddings.shape[0]), np.array(q_idxs))
            wait_list = np.setdiff1d(wait_list, low_density_idxs)
            random_selected = np.random.choice(wait_list, size=diff, replace=False)
            for i in range(random_selected.shape[0]):
                q_random_idxs.append(random_selected[i])
            q_idxs = np.concatenate((q_idxs, q_random_idxs), axis=0)

        """
        second stage data preparation: get the predicted ranking of unlabeled pool and send to second stage ranking learning
        """
        data_stage_II_idx = np.full(25000, -1)
        data_stage_II_rank = []

        # get density rankings
        cluster_densitys = []
        for i in range(n_cluster):
            tmp_density = torch.mean(torch.from_numpy(density_prob)[cluster_idxs == i])
            cluster_densitys.append(tmp_density)
        cluster_densitys = np.array(cluster_densitys).argsort()

        """
        prob_ranking part: using classification as data selection method
        """
        # k = 0
        # for i in range(n_cluster):
        #     '''
        #     get rank 0, the most valuable idxs
        #     '''
        #     tmp_cluster_idx = cluster_densitys[i]
        #     tmp_cluster = np.arange(embeddings.shape[0])[cluster_idxs == tmp_cluster_idx][torch.from_numpy(density_prob)[cluster_idxs == tmp_cluster_idx].argsort()]
        #     tmp_idx_ranking = torch.from_numpy(density_prob)[cluster_idxs == tmp_cluster_idx].argsort()
        #     num = int(seq_rank[i] * tmp_cluster.flatten().shape[0] / embeddings.shape[0])
        #     if num != 0 and num <= tmp_cluster.flatten().shape[0]:
        #         for j in range(num):
        #             data_stage_II_idx[k] = tmp_cluster.flatten()[tmp_idx_ranking[j]]
        #             data_stage_II_rank.append(0)
        #             k += 1
        #         '''
        #         make other rankings: counting the left numbers and send even labels to them
        #         '''
        #         left_numbers = tmp_cluster.flatten().shape[0] - num
        #         max_cls = 49-i
        #         other_ranking = sorted(generate_weighted_list(left_numbers, max_cls))
        #         other_rank_idx = 0
        #         for j in range(num, tmp_cluster.flatten().shape[0]):
        #             data_stage_II_idx[k] = tmp_cluster.flatten()[tmp_idx_ranking[j]]
        #             data_stage_II_rank.append(other_ranking[other_rank_idx])
        #             k += 1
        #             other_rank_idx += 1
        #
        #     elif num > tmp_cluster.flatten().shape[0]:
        #         for j in range(tmp_cluster.flatten().shape[0]):
        #             data_stage_II_idx[k] = tmp_cluster.flatten()[tmp_idx_ranking[j]]
        #             data_stage_II_rank.append(0)
        #             k += 1
        # """
        # annotate low density images with label 199
        # """
        # low_den_indices = np.where(np.isin(data_stage_II_idx, low_density_idxs))[0]
        #
        # data_stage_II_rank = np.delete(np.array(data_stage_II_rank), low_den_indices)
        # data_stage_II_idx = np.setdiff1d(data_stage_II_idx, low_density_idxs)
        #
        # tmp_low_den_rank = np.full(low_density_idxs.shape[0], 199)
        # data_stage_II_rank = np.concatenate((data_stage_II_rank, tmp_low_den_rank), axis=0)
        # data_stage_II_idx = np.concatenate((data_stage_II_idx, low_density_idxs), axis=0)

        """
        multiple density and cluster rank, using it to make regression prediction
        """
        k = 0
        for i in range(n_cluster):
            low_bonus = i + 1
            high_bonus = n_cluster + 3*i
            tmp_cluster_idx = cluster_densitys[i]
            tmp_cluster = np.arange(embeddings.shape[0])[cluster_idxs == tmp_cluster_idx]
            tmp_idx_ranking = torch.from_numpy(density_prob)[cluster_idxs == tmp_cluster_idx].argsort()
            cluster_length = tmp_cluster.shape[0]
            bonus = generate_list(low_bonus, high_bonus, cluster_length)
            for j in range(cluster_length):
                tmp_idx = tmp_cluster[tmp_idx_ranking[j]]
                tmp_bonus = bonus[j]
                tmp_score = density[tmp_idx] * (2*i+1) * np.log(tmp_bonus)
                data_stage_II_idx[k] = tmp_idx
                data_stage_II_rank.append(tmp_score)
                k += 1

        data_stage_II_idx = data_stage_II_idx[data_stage_II_idx != -1]
        data_stage_II_rank = np.log(np.array(data_stage_II_rank, dtype='float32') + 1)

        return unlabeled_idxs[np.array(q_idxs).flatten()], unlabeled_idxs[data_stage_II_idx], data_stage_II_rank

    def query_second_stage_version_II(self, n):
        # Tunables
        FIXED_THR = self.density_threshold
        PCT_FALLBACK = 20
        LOW2_QUOTA = 10  # originally set to 10

        # ε-greedy hyperparams
        EPS = 0.05      # 5% exploration
        BAND_MULT = 5   # explore from a near-top band

        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        pred_rank = self.predict_rank(unlabeled_data).numpy().squeeze(1)
        q_idxs = []
        
        # compute density (optional in query_second_stage)
        # density = self.get_density(unlabeled_data)
        # # low_density_idxs = np.where(density < 0.075)[0] # filter out low density patches!!
        # # Low-density bucket with robust fallbacks
        # low_density_idxs = np.where(density < FIXED_THR)[0]
        # if low_density_idxs.size == 0:
        #     thr = np.percentile(density, PCT_FALLBACK)
        #     low_density_idxs = np.where(density <= thr)[0]
        # if low_density_idxs.size == 0:
        #     order = np.argsort(density)
        #     low_density_idxs = order[:min(LOW2_QUOTA, order.size)]
                
        low_density_idxs = np.array([], dtype=int)   # no low-density filtering

        """
        Following code is used in regression prediction
        """
        sorted_indices = pred_rank.argsort()

        # ε-greedy near-top selection (no density)
        n_query = int(n)
        topN = min(n_query, sorted_indices.shape[0])
        n_explore = int(round(EPS * topN))
        n_explore = max(0, min(n_explore, topN))
        n_main = topN - n_explore

        # main picks: strict top
        for i in range(n_main):
            q_idxs.append(sorted_indices[i])

        # exploration: random from a near-top band
        band_hi = min(sorted_indices.shape[0], n_main + BAND_MULT * max(1, n_explore))
        candidates = sorted_indices[n_main:band_hi]
        if n_explore > 0 and candidates.size > 0:
            explore = np.random.choice(candidates, size=min(n_explore, candidates.size), replace=False)
            q_idxs = np.concatenate((np.array(q_idxs, dtype=int), explore), axis=0)
        else:
            q_idxs = np.array(q_idxs, dtype=int)

        q_idxs = np.setdiff1d(np.array(q_idxs), low_density_idxs)

        wait_list = np.setdiff1d(np.arange(unlabeled_idxs.shape[0]), np.array(q_idxs))
        wait_list = np.setdiff1d(wait_list, low_density_idxs)
        diff = n_query - q_idxs.shape[0]
        if diff > 0 and wait_list.size > 0:
            random_selected = np.random.choice(wait_list, size=min(diff, wait_list.size), replace=False)
            q_idxs = np.concatenate((np.array(q_idxs), random_selected), axis=0)

        # guarded low-density add-back in case empty list
        take = min(LOW2_QUOTA, low_density_idxs.size)
        if take > 0:
            random_selected_low_density = np.random.choice(low_density_idxs, size=take, replace=False)
            q_idxs = np.concatenate((np.array(q_idxs), random_selected_low_density), axis=0)

        """
        Following code is used in classification predict
        """
        # cont = 0
        # for i in range(200):
        #     tmp_q_idxs = np.argwhere(pred_rank == i)
        #     print(tmp_q_idxs)
        #     if len(tmp_q_idxs[0]) != 0:
        #         for i_dx in range(len(tmp_q_idxs[0])):
        #             q_idxs.append(tmp_q_idxs[0][i_dx])
        #         cont += len(tmp_q_idxs[0])
        #         if cont > 205:
        #             q_idxs = np.array(q_idxs)[:205]
        #             break

        """
        Originally designed code, used for region selection
        """
        # # get latent feature
        # # confident sample include
        # # get uncertainty
        # probs = self.predict_prob(unlabeled_data)
        # log_probs = torch.log(probs + 1e-15)
        # uncertainties = (probs * log_probs).sum(1)
        #
        # wsi_name = self.get_wsi_name(unlabeled_data)
        # x_loaction, y_loaction = self.get_location(unlabeled_data)
        # # latent feature
        # embeddings = self.get_embeddings(unlabeled_data)
        # embeddings_pca = embeddings.numpy()
        #
        # unlabel_density = self.get_density(unlabeled_data)
        #
        # labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        # l_embeddings = self.get_embeddings(labeled_data)
        # l_embeddings = l_embeddings.numpy()
        #
        # l_density = self.get_density(labeled_data)
        # l_wsi_name = self.get_wsi_name(labeled_data)
        # l_x_loaction, l_y_loaction = self.get_location(labeled_data)
        # sorted_list = torch.argsort(l_density, descending=True)
        # l_cls = self.cls(labeled_data)
        #
        # # build_region
        # region_list = []
        # auto_annotate = []
        # count = 0
        # while True:
        #     if sorted_list.shape[0] < 16 or count > int(n * 0.01):
        #         break
        #     idx = sorted_list[0]
        #     core_x_local = l_x_loaction[idx]
        #     core_y_local = l_y_loaction[idx]
        #     core_density = l_density[idx]
        #     core_wsi_name = l_wsi_name[idx]
        #     tmp_wsi_list = np.where(wsi_name == core_wsi_name)
        #     core_embeddings = l_embeddings[idx]
        #
        #     distance_list = []
        #     for region_candidate_idx in tmp_wsi_list[0]:
        #         tmp_x_local = x_loaction[region_candidate_idx]
        #         tmp_y_local = y_loaction[region_candidate_idx]
        #         tmp_uncertainty = uncertainties[region_candidate_idx]
        #         tmp_density = unlabel_density[region_candidate_idx]
        #         tmp_ul_embeddings = embeddings_pca[[region_candidate_idx]]
        #
        #         if abs(core_density - tmp_density) < 5e-2:
        #             tmp_distance = (abs(core_x_local - tmp_x_local) / 224 + abs(core_y_local - tmp_y_local) / 224) / 400
        #             tmp_diff_feature = core_embeddings.dot(np.squeeze(tmp_ul_embeddings)) / (
        #                     np.linalg.norm(core_embeddings) * np.linalg.norm(np.squeeze(tmp_ul_embeddings)))
        #             marco_distance = tmp_uncertainty + tmp_distance - tmp_diff_feature
        #             distance_list.append(marco_distance)
        #
        #     sorted_indices = sorted(range(len(distance_list)), key=lambda k: distance_list[k])
        #     idx_list = []
        #     for item in sorted_indices:
        #         idx_list.append(tmp_wsi_list[0][item])
        #     idx_list = idx_list[int(len(idx_list) * 0.99):]
        #     if len(idx_list) != 0:
        #         count += len(idx_list)
        #         region_list.append(np.array(idx_list))
        #         auto_annotate.append(np.array([l_cls[idx] for i in range(len(idx_list))]))
        #         sorted_list = np.setdiff1d(sorted_list, np.array(idx_list))
        #
        # q2_idxs = np.array(region_list).flatten()
        # q2_label = np.array(auto_annotate).flatten()

        return unlabeled_idxs[q_idxs], unlabeled_idxs, pred_rank

    def wsi_pred(self):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        pred_rank = self.predict_wsi_score(unlabeled_data).numpy().squeeze(1)
        return unlabeled_idxs, pred_rank

    def MIL(self):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings, density = self.get_mil(unlabeled_data)
        density = density.numpy().squeeze(1)
        embeddings = embeddings.numpy()
        return embeddings, density, unlabeled_idxs




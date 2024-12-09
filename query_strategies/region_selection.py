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


def query_second_stage(self, n):
    # choose core patch, sort by uncertainty
    # find unlabeled data with high value
    unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

    # latent feature
    embeddings = self.get_embeddings(unlabeled_data)
    embeddings_pca = embeddings.numpy()

    unlabel_density = self.get_density(unlabeled_data)

    # get uncertainty
    probs = self.predict_prob(unlabeled_data)
    log_probs = torch.log(probs + 1e-15)
    uncertainties = (probs * log_probs).sum(1)
    sorted_list = torch.argsort(uncertainties).numpy()

    wsi_name = self.get_wsi_name(unlabeled_data)
    x_loaction, y_loaction = self.get_location(unlabeled_data)

    # build_region
    region_list = []
    while True:
        if sorted_list.shape[0] < 64:
            break
        idx = sorted_list[0]
        core_x_local = x_loaction[idx]
        core_y_local = y_loaction[idx]
        tmp_wsi_name = wsi_name[idx]
        tmp_wsi_list = np.where(wsi_name == tmp_wsi_name)

        idx_list = []
        for region_candidate_idx in tmp_wsi_list[0]:
            tmp_x_local = x_loaction[region_candidate_idx]
            tmp_y_local = y_loaction[region_candidate_idx]
            if (core_x_local - tmp_x_local) ** 2 < (16 * 224) ** 2 and (core_y_local - tmp_y_local) ** 2 < (
                    16 * 224) ** 2:
                idx_list.append(region_candidate_idx)
        if len(idx_list) != 0:
            region_list.append(np.array(idx_list))
            sorted_list = np.setdiff1d(sorted_list, np.array(idx_list))

    region_diversity_uncertainty = []
    for region in region_list:
        # uncertainty
        tmp_uncertainty = 0
        diversity = 0
        for i in range(region.shape[0]):
            tmp_uncertainty += uncertainties[region[i]]
        tmp_uncertainty /= region.shape[0]

        # diversity
        # 双中心化
        if region.shape[0] > 3:
            tmp_embeddings_pca = embeddings_pca[region]
            D = np.sqrt(1 / np.sum(tmp_embeddings_pca ** 2, axis=0))
            D = np.diag(D)

            # 计算矩阵
            G = tmp_embeddings_pca - np.mean(tmp_embeddings_pca, axis=0) - np.mean(tmp_embeddings_pca, axis=1)[:,
                                                                           np.newaxis] + np.mean(tmp_embeddings_pca)

            # 计算双中心化矩阵
            tmp_embeddings_pca = -1 / 2 * np.dot(np.dot(D, G.T).T, D)

            diversity_matrix = pdist(tmp_embeddings_pca, metric='euclidean')
            diversity_matrix = np.square(diversity_matrix) / np.var(diversity_matrix)
            diversity = np.sum(diversity_matrix)

        # 加权计算uncertainty and diversity
        tmp_diversity_uncertainty = 2 * diversity / region.shape[0] + tmp_uncertainty
        region_diversity_uncertainty.append(tmp_diversity_uncertainty)

    # region selection
    sorted_indices = [i for i, _ in
                      sorted(enumerate(region_diversity_uncertainty), key=lambda x: x[1], reverse=True)]
    q_idxs = region_list[sorted_indices[0]]
    for idx in sorted_indices[1:-1]:
        q_idxs = np.concatenate((q_idxs, region_list[idx]), axis=0)
        if q_idxs.shape[0] > n:
            break

    # confident sample include
    labeled_idxs, labeled_data = self.dataset.get_labeled_data()
    l_embeddings = self.get_embeddings(labeled_data)
    l_embeddings = l_embeddings.numpy()

    l_density = self.get_density(labeled_data)
    l_wsi_name = self.get_wsi_name(labeled_data)
    l_x_loaction, l_y_loaction = self.get_location(labeled_data)
    sorted_list = torch.argsort(l_density, descending=True)
    l_cls = self.cls(labeled_data)

    # build_region
    region_list = []
    auto_annotate = []
    count = 0
    while True:
        if sorted_list.shape[0] < 16 or count > int(n * 0.01):
            break
        idx = sorted_list[0]
        core_x_local = l_x_loaction[idx]
        core_y_local = l_y_loaction[idx]
        core_density = l_density[idx]
        core_wsi_name = l_wsi_name[idx]
        tmp_wsi_list = np.where(wsi_name == core_wsi_name)
        core_embeddings = l_embeddings[idx]

        distance_list = []
        for region_candidate_idx in tmp_wsi_list[0]:
            tmp_x_local = x_loaction[region_candidate_idx]
            tmp_y_local = y_loaction[region_candidate_idx]
            tmp_uncertainty = uncertainties[region_candidate_idx]
            tmp_density = unlabel_density[region_candidate_idx]
            tmp_ul_embeddings = embeddings_pca[[region_candidate_idx]]

            if abs(core_density - tmp_density) < 5e-2:
                tmp_distance = (abs(core_x_local - tmp_x_local) / 224 + abs(core_y_local - tmp_y_local) / 224) / 400
                tmp_diff_feature = core_embeddings.dot(np.squeeze(tmp_ul_embeddings)) / (
                        np.linalg.norm(core_embeddings) * np.linalg.norm(np.squeeze(tmp_ul_embeddings)))
                marco_distance = tmp_uncertainty + tmp_distance - tmp_diff_feature
                distance_list.append(marco_distance)

        sorted_indices = sorted(range(len(distance_list)), key=lambda k: distance_list[k])
        idx_list = []
        for item in sorted_indices:
            idx_list.append(tmp_wsi_list[0][item])
        idx_list = idx_list[int(len(idx_list) * 0.99):]
        if len(idx_list) != 0:
            count += len(idx_list)
            region_list.append(np.array(idx_list))
            auto_annotate.append(np.array([l_cls[idx] for i in range(len(idx_list))]))
            sorted_list = np.setdiff1d(sorted_list, np.array(idx_list))

    q2_idxs = np.array(region_list).flatten()
    q2_label = np.array(auto_annotate).flatten()

    return unlabeled_idxs[q_idxs], unlabeled_idxs[q2_idxs], q2_label


def inti_wsi(self, n=50):
    unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
    embeddings = self.get_embeddings(unlabeled_data)
    embeddings_pca = embeddings.numpy()

    wsi_name_tmp = self.get_wsi_name(unlabeled_data).flatten().tolist()
    wsi_name = self.get_wsi_name(unlabeled_data)
    colors = self.get_color(unlabeled_data).numpy()
    tmp = []
    diversity = []
    for i in range(250):
        if i in wsi_name_tmp:
            tmp_wsi_list = np.where(wsi_name == i)
            tmp_color = colors[tmp_wsi_list[0]]
            r_mean, g_mean, b_mean, c_mean = np.mean(tmp_color, axis=0)
            r_std, g_std, b_std, c_std = np.std(tmp_color, axis=0)
            tmp.append([r_mean, g_mean, b_mean, c_mean, r_std, g_std, b_std, c_std])

            if tmp_wsi_list[0].shape[0] > 2:
                tmp_embeddings_pca = embeddings_pca[tmp_wsi_list[0]]
                diversity_matrix = pdist(tmp_embeddings_pca, metric='euclidean')
                diversity_matrix = np.square(diversity_matrix) / np.var(diversity_matrix)
                diversity.append(np.sum(diversity_matrix))
            else:
                diversity.append(0)

    diversity = np.array(diversity)

    cluster_learner = KMeans(n_clusters=5)
    cluster_learner.fit(tmp)
    cluster_idxs = cluster_learner.predict(tmp)

    save_npy = {"{}".format('all_sort'): diversity.argsort()}
    for i in range(5):
        save_npy["{}".format(i)] = np.arange(diversity.shape[0])[cluster_idxs == i][
            diversity[cluster_idxs == i].argsort()].flatten()
    print(save_npy)
    np.save("./active_learning/wsi_select.npy", save_npy)


def query_third_stage(self, n):
    unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

    # get latent feature
    embeddings = self.get_embeddings(unlabeled_data)
    embeddings = embeddings.numpy()
    pca = PCA(n_components=30)
    embeddings_pca = torch.from_numpy(pca.fit_transform(embeddings))

    density = self.get_density(unlabeled_data)

    # get uncertainty
    probs = self.predict_prob(unlabeled_data)
    log_probs = torch.log(probs + 1e-15)
    uncertainties = (probs * log_probs).sum(1)
    uncertainties = uncertainties.unsqueeze(1)

    # combined variations
    combined_var = torch.cat([density, embeddings_pca], dim=1)
    combined_var = torch.cat([combined_var, uncertainties], dim=1)

    gbdt = load('./gbdt_model.joblib')
    pred_rank = gbdt.predict(combined_var)
    q_idxs = pred_rank.argsort(axis=0).flatten()[:2150]

    # confident sample include
    # get uncertainty
    probs = self.predict_prob(unlabeled_data)
    log_probs = torch.log(probs + 1e-15)
    uncertainties = (probs * log_probs).sum(1)

    wsi_name = self.get_wsi_name(unlabeled_data)
    x_loaction, y_loaction = self.get_location(unlabeled_data)
    # latent feature
    embeddings = self.get_embeddings(unlabeled_data)
    embeddings_pca = embeddings.numpy()

    unlabel_density = self.get_density(unlabeled_data)

    labeled_idxs, labeled_data = self.dataset.get_labeled_data()
    l_embeddings = self.get_embeddings(labeled_data)
    l_embeddings = l_embeddings.numpy()

    l_density = self.get_density(labeled_data)
    l_wsi_name = self.get_wsi_name(labeled_data)
    l_x_loaction, l_y_loaction = self.get_location(labeled_data)
    sorted_list = torch.argsort(l_density, descending=True)
    l_cls = self.cls(labeled_data)

    # build_region
    region_list = []
    auto_annotate = []
    count = 0
    while True:
        if sorted_list.shape[0] < 16 or count > int(n * 0.01):
            break
        idx = sorted_list[0]
        core_x_local = l_x_loaction[idx]
        core_y_local = l_y_loaction[idx]
        core_density = l_density[idx]
        core_wsi_name = l_wsi_name[idx]
        tmp_wsi_list = np.where(wsi_name == core_wsi_name)
        core_embeddings = l_embeddings[idx]

        distance_list = []
        for region_candidate_idx in tmp_wsi_list[0]:
            tmp_x_local = x_loaction[region_candidate_idx]
            tmp_y_local = y_loaction[region_candidate_idx]
            tmp_uncertainty = uncertainties[region_candidate_idx]
            tmp_density = unlabel_density[region_candidate_idx]
            tmp_ul_embeddings = embeddings_pca[[region_candidate_idx]]

            if abs(core_density - tmp_density) < 5e-2:
                tmp_distance = (abs(core_x_local - tmp_x_local) / 224 + abs(core_y_local - tmp_y_local) / 224) / 400
                tmp_diff_feature = core_embeddings.dot(np.squeeze(tmp_ul_embeddings)) / (
                        np.linalg.norm(core_embeddings) * np.linalg.norm(np.squeeze(tmp_ul_embeddings)))
                marco_distance = tmp_uncertainty + tmp_distance - tmp_diff_feature
                distance_list.append(marco_distance)

        sorted_indices = sorted(range(len(distance_list)), key=lambda k: distance_list[k])
        idx_list = []
        for item in sorted_indices:
            idx_list.append(tmp_wsi_list[0][item])
        idx_list = idx_list[int(len(idx_list) * 0.99):]
        if len(idx_list) != 0:
            count += len(idx_list)
            region_list.append(np.array(idx_list))
            auto_annotate.append(np.array([l_cls[idx] for i in range(len(idx_list))]))
            sorted_list = np.setdiff1d(sorted_list, np.array(idx_list))

    q2_idxs = np.array(region_list).flatten()
    q2_label = np.array(auto_annotate).flatten()

    return unlabeled_idxs[q_idxs], unlabeled_idxs[q2_idxs], q2_label
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import torch.utils.data as d
from torchvision import transforms
from PIL import Image
import os


class second_stage_pool(d.Dataset):
    def __init__(self, data_list):
        self.imgs = data_list
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        img_path, cls, density, location, wsi_name, x, y, rank = self.imgs[idx]
        cls = torch.tensor(cls).item()
        img = Image.open(os.path.join('./labeled data', img_path)).convert('RGB')
        img = self.transform(img)
        # color_style = 0

        return img, cls, density, location, wsi_name, idx, x, y, 0, rank

    def __len__(self):
        return len(self.imgs)


class train_for_loss_pool(d.Dataset):
    def __init__(self, data_list):
        self.imgs = data_list
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        img_path, cls, density, location, wsi_name, x, y, loss = self.imgs[idx]
        cls = torch.tensor(cls).item()
        img = Image.open(os.path.join('./labeled data', img_path)).convert('RGB')
        img = self.transform(img)
        # color_style = 0

        return img, cls, density, location, wsi_name, idx, x, y, 0, loss

    def __len__(self):
        return len(self.imgs)


class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        pass

    def query_second_stage(self, n):
        pass

    def query_third_stage(self, n):
        pass

    def inti_wsi(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def update_cls(self, pos_idxs, new_cls):
        self.dataset.labeled_idxs[pos_idxs] = True
        self.dataset.X_train[pos_idxs][:, 1] = new_cls

    def train(self, rd):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data, rd)

    def train_for_second_stage(self, rd, idx, rank):
        second_stage_data = self.dataset.X_train[idx]
        rank = np.array(rank)
        new_data = np.zeros((second_stage_data.shape[0], second_stage_data.shape[1] + 1), dtype=object)
        for i, row in enumerate(second_stage_data):
            new_data[i, :-1] = second_stage_data[i, :]
            new_data[i, -1] = rank[i]
        new_data = second_stage_pool(new_data)
        self.net.stage_II_training(new_data, rd)

    def train_for_loss(self, rd, idx, loss):
        loss_data = self.dataset.X_train[idx]
        loss = np.array(loss)
        new_data = np.zeros((loss_data.shape[0], loss_data.shape[1] + 1), dtype=object)
        for i, row in enumerate(loss_data):
            new_data[i, :-1] = loss_data[i, :]
            new_data[i, -1] = loss[i]
        new_data = train_for_loss_pool(new_data)
        self.net.loss_training(new_data, rd)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_rank(self, data):
        pred_rank = self.net.predict_rank(data)
        return pred_rank

    def predict_wsi_score(self, data):
        predict_wsi_score = self.net.predict_wsi_score(data)
        return predict_wsi_score

    def predict_loss(self, data):
        pred_loss = self.net.predict_loss(data)
        return pred_loss

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs

    def get_img_uncertainty(self, data):
        uncertainty = self.net.uncertainty_pred(data)
        return uncertainty

    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

    def get_density(self, data):
        density = self.net.get_density(data)
        return density

    def get_location(self, data):
        x_location, y_location = self.net.get_location(data)
        return x_location, y_location

    def get_wsi_name(self, data):
        wsi_name = self.net.get_wsi_name(data)
        return wsi_name

    def get_cls(self, idx):
        tmp_data = self.dataset.X_train[idx]
        cls_count = torch.zeros(7)
        for i in range(tmp_data.shape[0]):
            cls = tmp_data[i, 1]
            cls_count[cls] += 1
        return cls_count

    def get_wsi(self, idx):
        tmp_data = self.dataset.X_train[idx]
        wsi_count = []
        for i in range(tmp_data.shape[0]):
            wsi_count.append(tmp_data[i, -3])

        counter = Counter(wsi_count)
        return counter

    def get_all_infor(self, idx):
        return self.dataset.X_train[idx]

    def get_color(self, data):
        color = self.net.get_color_style(data)
        return color

    def cls(self, data):
        cls = self.net.get_cls(data)
        return cls

    def get_mil(self, data):
        embeddings, density = self.net.get_mil(data)
        return embeddings, density

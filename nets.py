import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision.transforms as transforms
from dataloader import enhance_uncertainty
import matplotlib.pyplot as plt


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


def uncertainty_transformation(rd):
    if rd == 1:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    elif rd == 2:
        transform = transforms.Compose([
            transforms.RandomRotation(75),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    elif rd == 3:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    elif rd == 4:
        transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    elif rd == 5:
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    elif rd == 6:
        transform = transforms.Compose([
            transforms.GaussianBlur(3),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
    return transform


class Net:
    def __init__(self, net, params, device, root, net_stage_II, net_loss=None):
        self.net = net
        self.params = params
        self.device = device
        self.root = root
        self.net_stage_II = net_stage_II(num_classes=1).to(self.device)
        self.net_loss = net_loss

    def train(self, data, round):
        n_epoch = self.params['n_epoch']
        self.clf = self.net(num_classes=6).to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, density, location, wsi_name, idxs, a, b, color_style) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1, _ = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
        save_model_path = os.path.join(self.root, 'round_{}.pth'.format(round))
        torch.save(self.clf.state_dict(), save_model_path)

    def predict(self, data):
        self.clf.eval()
        acc = 0
        preds = torch.zeros(len(data), dtype=torch.int64)
        labels = torch.zeros(len(data), dtype=torch.int64)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1, _ = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
                labels[idxs] = y.cpu()
                acc += 1.0 * (y == pred).sum().item() / len(data)
        return preds, acc, labels

    def stage_II_training(self, data, round):
        n_epoch = 9
        self.net_stage_II.train()
        self.clf.eval()
        reg_loss = torch.nn.MSELoss()
        optimizer = optim.SGD(self.net_stage_II.parameters(), lr=0.0025, momentum=0.9)
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        loss_record = []
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for batch_idx, (x, y, density, location, wsi_name, idxs, a, b, color_style, rank) in enumerate(loader):
                x, rank = x.to(self.device), rank.to(self.device)
                out, e1, features = self.clf(x)
                optimizer.zero_grad()
                reg_result = self.net_stage_II(features)
                reg_result = reg_result.squeeze(1)

                """
                here is classification prob, ce loss
                """
                # cls_result = self.net_stage_II(features)
                # loss = F.cross_entropy(cls_result, rank)

                # """
                # here is regression prob, mse loss
                # """
                loss = reg_loss(reg_result, rank)
                loss_record.append(loss.detach().cpu().numpy())

                """
                here is rank pred loss 
                """
                # loss = LossPredLoss(reg_result, rank)
                # loss_record.append(loss.detach().cpu().numpy())

                """
                here is combination for two loss
                """
                # loss1 = reg_loss(reg_result, rank)
                # loss2 = LossPredLoss(reg_result, rank)
                # loss = 0.5*loss2 + 0.5*loss1
                # loss_record.append(loss.detach().cpu().numpy())

                print("loss_{}".format(loss))
                loss.backward()
                # clipping
                torch.nn.utils.clip_grad_norm_(self.net_stage_II.parameters(), max_norm=1.0)
                optimizer.step()
        print(loss_record)
        plt.plot(loss_record)
        if not os.path.exists(os.path.join(self.root, 'loss')):
            os.mkdir(os.path.join(self.root, 'loss'))
        plt.savefig(os.path.join(self.root, 'loss', 'loss_{}.png'.format(round)))
        save_model_path = os.path.join(self.root, 'stage_II_round_{}.pth'.format(round))
        torch.save(self.net_stage_II.state_dict(), save_model_path)

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), 6])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1, _ = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_rank(self, data):
        self.net_stage_II.eval()
        self.clf.eval()
        pred_rank = torch.zeros([len(data), 1])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1, features = self.clf(x)
                tmp_rank = self.net_stage_II(features)
                """
                used for rank prob, ce loss; if regression, just annotate it
                """
                # tmp_rank = tmp_rank.max(1)[1]
                pred_rank[idxs] = tmp_rank.cpu()
        return pred_rank

    def predict_wsi_score(self, data):
        self.clf = self.net(num_classes=6).to(self.device)
        clf_path = './active_learning/exp/methodMY exp_MY_stage_II_seed_35_res101_module_simple_run_9_log_score_mse_loss/round_6.pth'
        net_II_path = './active_learning/exp/methodMY exp_MY_stage_II_seed_35_res101_module_simple_run_9_log_score_mse_loss/stage_II_round_1.pth'
        self.clf.load_state_dict(torch.load(clf_path, map_location='cuda:1'))
        self.net_stage_II.load_state_dict(torch.load(net_II_path, map_location='cuda:1'))
        self.net_stage_II.eval()
        self.clf.eval()
        pred_rank = torch.zeros([len(data), 1])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1, features = self.clf(x)
                tmp_rank = self.net_stage_II(features)
                pred_rank[idxs] = tmp_rank.cpu()
        return pred_rank

    def predict_loss(self, data):
        self.net_loss.eval()
        self.clf.eval()
        pred_loss = torch.zeros([len(data), 1])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1, features = self.clf(x)
                tmp_loss = self.net_loss(features)
                pred_loss[idxs] = tmp_loss.cpu()
        return pred_loss

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), 6])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1, _ = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def uncertainty_pred(self, data):
        self.clf.train()
        pred_prob = torch.zeros([len(data), 7, 6])
        uncertainty_pred = torch.zeros([len(data), 1], dtype=torch.double)
        for count in range(7):
            tmp_data = enhance_uncertainty(data, transform=uncertainty_transformation(count))
            loader = DataLoader(tmp_data, shuffle=False, **self.params['test_args'])
            with torch.no_grad():
                for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1, _ = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    pred_prob[idxs, count, :] = prob.cpu()
        var = torch.var(pred_prob, dim=1)
        uncertainty = torch.sum(var, dim=1)
        uncertainty_pred[:, 0] = uncertainty
        return uncertainty_pred

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), 6])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1, _ = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1, _ = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings

    def get_mil(self, data):
        self.clf = self.net(num_classes=6).to(self.device)
        clf_path = './active_learning/exp/methodMY exp_MY_stage_II_seed_35_res101_module_simple_run_9_log_score_mse_loss/round_6.pth'
        self.clf.load_state_dict(torch.load(clf_path, map_location='cuda:1'))
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        densitys = torch.zeros([len(data), 1], dtype=torch.double)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, density, location, wsi_name, idxs, a, b, color_style in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1, _ = self.clf(x)
                embeddings[idxs] = e1.cpu()
                densitys[idxs] = torch.unsqueeze(torch.tensor(density).cpu(), dim=1)
        return embeddings, densitys

    def get_density(self, data):
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        densitys = torch.zeros([len(data), 1], dtype=torch.double)
        for img, cls, density, location, wsi_name, idxs, x, y, color_style in loader:
            densitys[idxs] = torch.unsqueeze(torch.tensor(density).cpu(), dim=1)
        return densitys

    def get_wsi_name(self, data):
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        wsi = np.zeros([len(data), 1])
        for img, cls, density, location, wsi_name, idx, x, y, color_style in loader:
            wsi[idx] = torch.unsqueeze(torch.tensor(wsi_name).cpu(), dim=1)
        return wsi

    def get_location(self, data):
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        x_location = torch.zeros([len(data), 1], dtype=torch.long)
        y_location = torch.zeros([len(data), 1], dtype=torch.long)
        for img, cls, density, location, wsi_name, idx, x, y, color_style in loader:
            x_location[idx] = torch.unsqueeze(torch.tensor(x).cpu(), dim=1)
            y_location[idx] = torch.unsqueeze(torch.tensor(y).cpu(), dim=1)
        return x_location, y_location

    def get_cls(self, data):
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        clss = torch.zeros([len(data), 1], dtype=torch.long)
        for img, cls, density, location, wsi_name, idx, x, y, color_style in loader:
            clss[idx] = torch.unsqueeze(torch.tensor(cls).cpu(), dim=1)
        return clss

    def get_color_style(self, data):
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        color = torch.zeros([len(data), 4])
        for img, cls, density, location, wsi_name, idx, x, y, color_style in loader:
            color[idx] = color_style.cpu()
        return color
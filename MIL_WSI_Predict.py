import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 数据集类
class WSIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        for label_folder in os.listdir(root_dir):
            label = int(label_folder)  # 假设文件夹名称可以直接转换为整数标签
            folder_path = os.path.join(root_dir, label_folder)
            for file in os.listdir(folder_path):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(folder_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        features = np.load(file_path)[:, :50]
        return torch.tensor(features, dtype=torch.float32), int(label) - 1


# 注意力层
class BiDirectionalAttention(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super(BiDirectionalAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_heads)])

    def forward(self, x):
        weights = [torch.softmax(att_head(x), dim=0) for att_head in self.attention_heads]
        weighted_features = [weights[i] * x for i in range(self.num_heads)]
        aggregated_features = [wf.sum(1) for wf in weighted_features]
        return torch.cat(aggregated_features, dim=1), weights


class MILAttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=1):
        super(MILAttentionModel, self).__init__()
        self.attention = BiDirectionalAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(0.8)
        self.batch_norm = nn.BatchNorm1d(input_dim * num_heads)
        self.classifier = nn.Linear(input_dim * num_heads, num_classes)

    def forward(self, bag):
        weighted_bag, weights = self.attention(bag)
        weighted_bag = self.dropout(weighted_bag)
        weighted_bag = self.batch_norm(weighted_bag)
        return self.classifier(F.relu(weighted_bag)), weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.5, gamma=2, num_classes=6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return F_loss.mean()


for method in ['Ours', 'Random']:
    print(method)

    for r in range(12):
        dims = [50, 100, 150, 200, 250]
        accs = []
        aucs = []
        f1_scores = []

        for dim in dims:
            dataset = WSIDataset(
                root_dir=r'.\MIL_new\selected_results\selected_train_set\{}_selected\{}'.format(dim, method))
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            test_dataset = WSIDataset(
                root_dir=r'.\MIL_new\selected_results\selected_test_set\{}_selected_test\{}'.format(
                    dim, method))
            test_dataloader = DataLoader(test_dataset, batch_size=39, shuffle=False)

            # 模型初始化
            input_dim = 50
            num_classes = 6
            model = MILAttentionModel(input_dim, num_classes).to(device)
            criterion = FocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0075)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

            # 训练循环
            if method == 'Ours':
                num_epochs = 1500  # 定义迭代次数
            else:
                num_epochs = 800

            loss_values = []  # 用于存储每个epoch的损失值
            acc = []
            test_accuracy_values = []
            e = []

            best_acc = 0
            best_f1 = 0
            best_auc = 0

            for epoch in tqdm(range(num_epochs)):
                correct = 0
                total = 0
                total_loss = 0
                for features, label in dataloader:
                    optimizer.zero_grad()
                    features, label = features.to(device), label.to(device)
                    outputs, _ = model(features)
                    loss = criterion(outputs, label)

                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    # 预测和准确率计算
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label.cuda()).sum().item()

                average_loss = total_loss / len(dataloader)
                accuracy = 100 * correct / total

                scheduler.step()

                model.eval()
                total_preds = []
                total_labels = []
                correct = 0
                total = 0

                with torch.no_grad():
                    for features, label in test_dataloader:
                        features, label = features.to(device), label.to(device)
                        outputs, _ = model(features)
                        _, predicted = torch.max(outputs.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()
                        total_preds.extend(predicted.cpu().numpy())
                        total_labels.extend(label.cpu().numpy())

                test_accuracy = correct / total
                test_f1 = f1_score(total_labels, total_preds, average='macro')
                test_auc = roc_auc_score(total_labels, F.softmax(outputs, dim=1).cpu().numpy(), multi_class='ovr')

                if test_accuracy >= best_acc:
                    best_acc = test_accuracy
                    if test_f1 > best_f1:
                        best_f1 = test_f1
                    if test_auc > best_auc:
                        best_auc = test_auc

                if epoch % 500 == 0:
                    acc.append(accuracy)
                    loss_values.append(average_loss)
                    test_accuracy_values.append(test_accuracy)
                    e.append(epoch)
                    print(
                        f"{method} selected {dim} Epoch {epoch}, Loss: {average_loss}, best_acc: {best_acc}, best auc: {best_auc}, best_f1: {best_f1}")

            accs.append(best_acc)
            aucs.append(best_auc)
            f1_scores.append(best_f1)

        print(f'{method} round: {r}, acc: {accs}\n aucs: {aucs}\n f1_scores: {f1_scores}\n')

        with open(r'.\records\{}_accs.txt'.format(method), 'a') as file:
            file.write(f'{accs}\n')

        with open(r'.\records\{}_aucs.txt'.format(method), 'a') as file:
            file.write(f'{aucs}\n')

        with open(r'.\records\{}_f1_scores.txt'.format(method), 'a') as file:
            file.write(f'{f1_scores}\n')

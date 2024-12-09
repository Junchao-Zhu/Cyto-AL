import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch


# def get_label(file):
#     wsi_name = file.split("_")[0]+'.npy'
#     label_path = os.path.join('/home/yx/data/dict_classification', wsi_name)
#     tmp = np.load(label_path, allow_pickle=True).item()
#     return tmp.get(file)[0]

def get_label(file):
    for d in os.listdir('./dataset_final_6'):
        if file in os.listdir(os.path.join('./dataset_final_6', d)):
            cls = int(d) - 1
            return cls


def get_density(file):
    img_pred = Image.open(file)
    tmp = np.sum(np.array(img_pred))/ (224*224*255)
    return tmp


def get_location(file):
    tmp = file.split("_")
    if tmp[1] != 'row' and tmp[1] != 'x':
        x = int(tmp[1])
        y = int(tmp[2])
    else:
        x = int(tmp[2])
        y = int(tmp[4][:-4])
    location = (x, y)
    return location


def get_wsi_name(file):
    return file.split("_")[0]


def calculate_color_statistics(image):
    # 将图像转换为NumPy数组
    rgb_array = np.array(image)
    # 计算每个颜色通道的均值和标准差
    r_mean, g_mean, b_mean = np.mean(rgb_array, axis=(0, 1))
    grayscale_image = image.convert('L')
    # 将图像转换为NumPy数组
    grayscale_array = np.array(grayscale_image)
    # 计算灰度图像的对比度
    contrast = np.std(grayscale_array)
    my_list = [r_mean, g_mean, b_mean, contrast]
    result = torch.tensor(my_list, dtype=torch.float32)
    return result


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.png')]
    data_list = []
    count = 0
    for img_name in tqdm(img_list):
        img_path = os.path.join('img', img_name + '.png')
        cls = get_label(img_name + '.png')
        den = get_density(img_name + '.png')
        location = get_location(img_name + '.png')
        wsi_name = get_wsi_name(img_name + '.png')
        data_list.append((img_path,
                          cls,
                          den,
                          location,
                          wsi_name
                          ))
        count += 1
    return data_list


class basic_pool(data.Dataset):
    def __init__(self, data_list):
        self.imgs = data_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])

    def __getitem__(self, idx):
        img_path, cls, density, location, wsi_name, x, y = self.imgs[idx]

        img = Image.open(os.path.join('./labeled data', img_path)).convert('RGB')
        img = self.transform(img)

        return img, cls, density, location, wsi_name, idx, x, y, 0

    def __len__(self):
        return len(self.imgs)


class enhance_uncertainty(data.Dataset):
    def __init__(self, data_list, transform=None):
        self.imgs = data_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path, cls, density, location, wsi_name, x, y = self.imgs[idx]
        cls = torch.tensor(cls).item()
        img = Image.open(os.path.join('./labeled data', img_path)).convert('RGB')
        img = self.transform(img)

        return img, cls, density, location, wsi_name, idx, x, y, 0

    def __len__(self):
        return len(self.imgs)


class Data:
    def __init__(self, X_train, X_test, handler):
        self.X_train = X_train
        self.X_test = X_test
        self.handler = handler
        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        tmp_idxs = np.arange(self.n_pool)
        tmp_idxs = np.random.choice(tmp_idxs, size=num, replace=False)
        self.labeled_idxs[tmp_idxs] = True

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs])

    def get_enhance_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.X_train[unlabeled_idxs]

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train)

    def get_test_data(self):
        return self.handler(self.X_test)

    def second_stage_data(self):
        return


def get_data(handler):
    labeled_train = np.load('./data_infor/train_label_new_pred.npy', allow_pickle=True)
    labeled_test = np.load('./data_infor/test_label_new_pred.npy', allow_pickle=True)
    idxs = np.random.choice(np.arange(labeled_train.shape[0]), size=2000, replace=False)
    return Data(labeled_train, labeled_test, handler)


def wsi_img(handler, label_path):
    labeled_train = np.load(label_path, allow_pickle=True)
    name = label_path.split('_')[-1][:-4]
    labeled_test = np.load('./data_infor/test_label_new_pred.npy', allow_pickle=True)
    return Data(labeled_train, labeled_test, handler), name


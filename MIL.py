import argparse
import os
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy, get_handler
from pprint import pprint
from dataloader import wsi_img

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=0, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=2000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=8, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="WSI", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="MY",
                    choices=["RandomSampling",
                             "LeastConfidence",
                             "MarginSampling",
                             "EntropySampling",
                             "LeastConfidenceDropout",
                             "MarginSamplingDropout",
                             "EntropySamplingDropout",
                             "KMeansSampling",
                             "KCenterGreedy",
                             "BALDDropout",
                             "AdversarialBIM",
                             "AdversarialDeepFool", "learn_for_loss"], help="query strategy")

args = parser.parse_args()
pprint(vars(args))

for file in os.listdir('./active_learning/data_infor/wsis'):
    print(file)
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    print(device)

    acc_count = []
    cls_table = []
    wsi_name = []

    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    dataset, name = wsi_img(get_handler(args.dataset_name), os.path.join('./data_infor/wsis', file))                   # load dataset
    net = get_net(args.strategy_name, device, 'all_result')       # load network
    strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    embeddings, density, unlabeled_idxs = strategy.MIL()
    wsi_score, wsi_idx = strategy.wsi_pred()

    np.save('./MIL_new/embedding_{}.npy'.format(name), embeddings)
    np.save('./MIL_new/density_{}.npy'.format(name), density)
    np.save('./MIL_new/unlabeled_idx_round_{}.npy'.format(name), unlabeled_idxs)
    np.save('./MIL_new/wsi_score_{}.npy'.format(name), wsi_score)
    np.save('./MIL_new/wsi_idx_{}.npy'.format(name), wsi_idx)




import argparse
import os
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=25, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=1000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=2000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=8, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="MY", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
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

exp = '_{}_{}_resnet152'.format(args.strategy_name, args.seed)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

acc_count = []
cls_table = []
wsi_name = []

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

all_result = './exp/method{} exp{}'.format(args.strategy_name, exp)
if not os.path.exists(all_result):
    os.mkdir(all_result)

train_log_filepath = os.path.join(all_result, "train_log_{}_seed_{}.txt".format(args.strategy_name, args.seed))

to_write = 'strategy_name:{} n_init_labeled:{} n_round:{} n_query:{}\n'.format(args.strategy_name, args.n_init_labeled, args.n_round, args.n_query)
with open(train_log_filepath, "a") as f:
    f.write(to_write)

dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device, all_result)       # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train('0')

# preds, acc = strategy.predict(dataset.get_test_data())

preds, acc, labels = strategy.predict(dataset.get_test_data())
np.save('./{}/stage_I_pred_result_round_{}.npy'.format(all_result, '0'), preds)
np.save('./{}/stage_I_labels_round_{}.npy'.format(all_result, '0'), labels)

acc_count.append(acc)
print(f"Round 0 testing accuracy: {acc}")

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")
    if rd < 8:
        # query_idxs = strategy.query(args.n_query)
        query_idxs, train_stage_two_idx, stage_II_rank = strategy.query(args.n_query)
        # np.save('./{}/stage_I_score_round_{}.npy'.format(all_result, rd), stage_II_rank)
        # np.save('./{}/stage_I_unlabeled_idx_round_{}.npy'.format(all_result, rd), train_stage_two_idx)
        # strategy.train_for_second_stage(rd, train_stage_two_idx, stage_II_rank)
    # query
    else:
        query_idxs, unlabeled_idx, pred_score = strategy.query_second_stage_version_II(args.n_query)
        np.save('./{}/stage_I_pred_score_round_{}.npy'.format(all_result, rd), pred_score)
        np.save('./{}/stage_I_pred_unlabeled_idx_round_{}.npy'.format(all_result, rd), unlabeled_idx)
        # strategy.update_cls(re_label_idx, re_cls)
    cls_count = strategy.get_cls(query_idxs)
    print('MY class_count:{}'.format(cls_count))
    cls_table.append(cls_count)

    wsi = strategy.get_wsi(query_idxs)
    print('MY wsi_count:{}'.format(wsi))
    wsi_name.append(wsi)
    all_infor = strategy.get_all_infor(query_idxs)
    print(all_infor)
    all_infor_save = os.path.join(all_result, 'round_{}_infor.npy'.format(rd))
    np.save(all_infor_save, all_infor)

    to_write = "strategy_name {} Round {} Acc {} Cls_count {} Wsi_count {}\n".format(args.strategy_name, rd, acc, cls_count, wsi)

    with open(train_log_filepath, "a") as f:
        f.write(to_write)

    # update labels
    strategy.update(query_idxs)
    strategy.train(rd)

    # calculate accuracy
    # preds, acc = strategy.predict(dataset.get_test_data())

    preds, acc, labels = strategy.predict(dataset.get_test_data())
    np.save('./{}/stage_I_pred_result_round_{}.npy'.format(all_result, rd), preds)
    np.save('./{}/stage_I_labels_round_{}.npy'.format(all_result, rd), labels)
    acc_count.append(acc)
    print(f"Round {rd} MY_testing accuracy: {acc}")

print("acc_count:{}".format(acc_count))
print('cls_table:{}'.format(cls_table))
print('wsi_count:{}'.format(wsi_name))

to_write = "strategy_name {} Acc {} cls_table {} wsi_count {}\n".format(args.strategy_name, acc_count, cls_table, wsi_name)

with open(train_log_filepath, "a") as f:
    f.write(to_write)
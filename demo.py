import argparse
import os
import numpy as np
import torch
import csv
from dataloader import make_dataset
from utils import get_dataset, get_net, get_strategy, _find_inner_module, _load_state_dict_safely, _now, _mins
from pprint import pprint
from datetime import datetime
from sklearn.model_selection import train_test_split

#dataset_path = '/data1/yx/data/dataset_final_6'
train_path = './data_infor/train_label_new_pred.npy'
test_path  = './data_infor/test_label_new_pred.npy'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default="/data1/yx/data/dataset_final_6", help="Path to dataset")
parser.add_argument('--seed', type=int, default=25, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=1000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=2000, help="number of queries per round") # increase by n_query every round
#parser.add_argument('--n_round', type=int, default=5, help="number of rounds")
parser.add_argument('--n_round', type=int, default=6, help="number of rounds")
# parser.add_argument('--load_pth', type=str, default='', help='Path to .pth to initialize weights before round 0')
parser.add_argument('--ranking_learning_module_start_round', type=int, default=6, help="Change this to modify after which round to use only the faster trained Ranking Learning Module for prediction")
parser.add_argument('--imagenet_backbone', action='store_true', help='If want to use a pretrained resnet backbone as initial guidance for the AL process...Initialize the classifier backbone from ImageNet ResNet152')
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

all_result = './exp/method{} exp{}'.format(args.strategy_name, exp)

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device, all_result)       # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

net.use_imagenet_backbone = args.imagenet_backbone

# if want to load a previous trained segmentation model:
# if getattr(args, "load_pth", ""):
#     module, path = _find_inner_module(net)
#     if module is None:
#         # Show what we can see to quickly diagnose
#         print("[init] Could not find an inner nn.Module. Attributes with state_dict():",
#               [n for n,v in vars(net).items() if hasattr(v, "state_dict")])
#         raise AttributeError("No inner nn.Module found on the wrapper; "
#                              "please tell me the attribute name and I’ll adapt the loader.")
#     print(f"[init] found inner module at path: {path} -> {type(module).__name__}")
#     _load_state_dict_safely(module, args.load_pth, ignore_head=True)


def _write_csv_row(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)

# pull hyperparams from the strategy (with safe fallbacks)
try:
    k_cluster, density_threshold, num_rank_sample = strategy.get_hparams()
except Exception:
    k_cluster         = getattr(strategy, "k_cluster", 50)
    density_threshold = float(getattr(strategy, "density_threshold", 0.075))
    num_rank_sample   = getattr(strategy, "num_ranking_sample", 2050)

# timestamp like 2025-08-14_13-07-42
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _fmt_thr(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", "p")

round_tag = f"R{args.n_round}"
hp_tag = f"k{int(k_cluster)}_thr{_fmt_thr(density_threshold)}_low{int(num_rank_sample)}"

os.makedirs(all_result, exist_ok=True)

time_csv = os.path.join(all_result, f"cytoAL_time_{stamp}_{round_tag}_{hp_tag}.csv")
acc_csv  = os.path.join(all_result, f"cytoAL_acc_{stamp}_{round_tag}_{hp_tag}.csv")

# write headers once
_time_hdr = ["round", "query_sec", "train_sec", "test_sec", "total_sec"]
_acc_hdr  = ["round", "avg_acc", "I", "II", "III", "IV", "V", "VI"]

# times_csv = os.path.join(all_result, "times_per_round.csv")
# with open(times_csv, "w", newline="") as f:
#     csv.writer(f).writerow(["round", "query_sec", "train_sec", "test_sec", "total_sec"])

if not os.path.exists('./exp'):
    os.mkdir('./exp')
if not os.path.exists(all_result):
    os.mkdir(all_result)

train_log_filepath = os.path.join(all_result, "train_log_{}_seed_{}.txt".format(args.strategy_name, args.seed))

to_write = 'strategy_name:{} n_init_labeled:{} n_round:{} n_query:{}\n'.format(args.strategy_name, args.n_init_labeled, args.n_round, args.n_query)
with open(train_log_filepath, "a") as f:
    f.write(to_write)

# Make dataset
dataset_path = args.dataset_path
data_list = np.array(make_dataset(dataset_path), dtype=object)
np.save('data_list.npy', data_list, allow_pickle=True)

# Split into training and testing data sets .npy files (train:test = 8:2)
full_list = np.load('data_list.npy', allow_pickle=True)
train_list, test_list = train_test_split(
    full_list,
    test_size = 0.2,
    train_size = 0.8,
    random_state=25,
    shuffle=True
)
os.makedirs('./data_infor', exist_ok=True)
for path in (train_path, test_path):
    if os.path.isfile(path):
        os.remove(path)
np.save(train_path, train_list, allow_pickle=True)
np.save(test_path,  test_list,  allow_pickle=True)


def per_class_accuracy(labels, preds, n_classes=6, class_names=("I","II","III","IV","V","VI")):
    if torch.is_tensor(labels): labels = labels.cpu().numpy()
    if torch.is_tensor(preds):  preds  = preds.cpu().numpy()

    accs = []
    for c in range(n_classes):
        mask = (labels == c)
        total = mask.sum()
        correct = (preds[mask] == c).sum()
        accs.append(float(correct) / total if total > 0 else float("nan"))
    # pretty print
    pretty = " ".join(f"{name}:{a:.4f}" if a==a else f"{name}:NA"  # NA if no samples of a class this round
                      for name, a in zip(class_names, accs))
    return accs, pretty

def counts_per_class(y, n_classes=6):
    if torch.is_tensor(y): y = y.cpu().numpy()
    y = y.astype(int)
    cnt = np.bincount(y, minlength=n_classes)
    pretty = " ".join(f"{i+1}:{cnt[i]}" for i in range(n_classes))
    return pretty, cnt

def labeled_pool_counts(dataset, n_classes=6):
    # boolean mask of which training items are labeled right now
    mask = np.asarray(dataset.labeled_idxs).astype(bool)
    idxs = np.arange(dataset.n_pool)[mask]

    # extract the class label from each (img_path, cls, density, location, wsi_name, x, y)
    raw = [dataset.X_train[i][1] for i in idxs]

    # normalize to integer class ids
    ids = []
    for y in raw:
        # torch scalar -> int
        if hasattr(y, "item"):
            try:
                ids.append(int(y.item())); continue
            except Exception:
                pass
        # list/tuple/np array -> take argmax if vector-like, else first element
        if isinstance(y, (list, tuple)):
            try:
                ids.append(int(np.argmax(np.asarray(y)))); continue
            except Exception:
                ids.append(int(y[0])); continue
        try:
            ids.append(int(y))
        except Exception:
            ids.append(int(np.argmax(np.asarray(y))))

    ids = np.asarray(ids, dtype=int)
    cnt = np.bincount(ids, minlength=n_classes)
    pretty = " ".join(f"{i+1}:{cnt[i]}" for i in range(n_classes))
    return pretty, cnt

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

### BEGIN ROUND 0 ###
print("Round 0")
# ---- TRAIN round 0 ----
round0_t0 = _now()
t0 = _now()
strategy.train('0')
train0_sec = _now() - t0

# ---- TEST round 0 ----
t0 = _now()
# preds, acc = strategy.predict(dataset.get_test_data())
preds, acc, labels = strategy.predict(dataset.get_test_data())
test0_sec = _now() - t0
round0_sec = _now() - round0_t0
np.save('./{}/stage_I_pred_result_round_{}.npy'.format(all_result, '0'), preds)
np.save('./{}/stage_I_labels_round_{}.npy'.format(all_result, '0'), labels)
acc_count.append(acc)

print(f"Round 0 time -> query: {_mins(0.0)} | train: {_mins(train0_sec)} | test: {_mins(test0_sec)} | total: {_mins(round0_sec)}")
# with open(times_csv, "a", newline="") as f:
#     csv.writer(f).writerow([0, f"{0.0:.3f}", f"{train0_sec:.3f}", f"{test0_sec:.3f}", f"{round0_sec:.3f}"])

print(f"Round 0 testing accuracy: {acc}")

pcs, pretty = per_class_accuracy(labels, preds)
print(f"Round 0 per-class accuracy -> {pretty}")

# pretty, test_counts = counts_per_class(labels, n_classes=6)
# print(f"Round 0 test set sample count per label -> {pretty}")

pretty, cnt_lab0 = labeled_pool_counts(dataset, n_classes=6)
print(f"Round 0 labeled pool -> {pretty} (total {cnt_lab0.sum()})")

_write_csv_row(time_csv, _time_hdr,
               [0, f"{0.0:.3f}", f"{train0_sec:.3f}", f"{test0_sec:.3f}", f"{round0_sec:.3f}"])

_write_csv_row(acc_csv, _acc_hdr,
               [0, f"{acc:.6f}"] + [f"{x:.6f}" if x==x else "NA" for x in pcs])
### END ROUND 0 ###

### BEGIN MAIN LOOP ###
EPOCHS_BEFORE = 50
EPOCHS_AFTER  = 10 # reduce training time after ranking learning module is trained in previous rounds
for rd in range(1, args.n_round + 1):
    print(f"Round {rd}")
    rd_t0 = _now()

    # ---------- QUERY ----------
    t0 = _now()
    if rd < args.ranking_learning_module_start_round:
        pretty_lab, cnt_lab = labeled_pool_counts(dataset, n_classes=6)
        print(f"Round {rd} labeled pool -> {pretty_lab} (total {cnt_lab.sum()})")
        # query_idxs = strategy.query(args.n_query)
        query_idxs, train_stage_two_idx, stage_II_rank = strategy.query(args.n_query)
        # np.save(f'./{all_result}/stage_I_score_round_{rd}.npy', stage_II_rank)
        # np.save(f'./{all_result}/stage_I_unlabeled_idx_round_{rd}.npy', train_stage_two_idx)

        # Train the Ranking Learning Module, once trained, can probably use it for prediction the subsequent tests (i.e. go directly to the else statement in round 0)
        # The Ranking Learning Module should speed up the process of picking the next batch, i.e. choosing/learning which unlabeled patches should be labeled next and move into the training set
        strategy.train_for_second_stage(rd, train_stage_two_idx, stage_II_rank)
    else:
        query_idxs, unlabeled_idx, pred_score = strategy.query_second_stage_version_II(args.n_query)
        np.save('./{}/stage_I_pred_score_round_{}.npy'.format(all_result, rd), pred_score)
        np.save('./{}/stage_I_pred_unlabeled_idx_round_{}.npy'.format(all_result, rd), unlabeled_idx)
        # strategy.update_cls(re_label_idx, re_cls)  # optional label correction / relabeling
    query_sec = _now() - t0

    # Bookkeeping prints/saves
    cls_count = strategy.get_cls(query_idxs)
    print('MY class_count:{}'.format(cls_count))
    cls_table.append(cls_count)

    wsi = strategy.get_wsi(query_idxs)
    print('MY wsi_count:{}'.format(wsi))
    wsi_name.append(wsi)
    all_infor = strategy.get_all_infor(query_idxs)
    print(all_infor)
    np.save(os.path.join(all_result, f'round_{rd}_infor.npy'), all_infor)

    ### UPDATE + TRAIN ###
    # set max epochs for this round
    epochs_this = EPOCHS_AFTER if rd >= args.ranking_learning_module_start_round else EPOCHS_BEFORE
    if hasattr(strategy, "net") and hasattr(strategy.net, "params"):
        strategy.net.params["n_epoch"] = int(epochs_this)
    else:
        # fallback in case Net is called directly
        net.params["n_epoch"] = int(epochs_this)

    t0 = _now()
    strategy.update(query_idxs)
    strategy.train(rd) # train
    train_sec = _now() - t0

    ### TEST / PREDICT ###
    t0 = _now()
    preds, acc, labels = strategy.predict(dataset.get_test_data())
    test_sec = _now() - t0
    total_sec = _now() - rd_t0

    # Save predictions/labels
    np.save(f'./{all_result}/stage_I_pred_result_round_{rd}.npy', preds)
    np.save(f'./{all_result}/stage_I_labels_round_{rd}.npy', labels)
    acc_count.append(acc)

    # Logs
    print(f"Round {rd} MY_testing accuracy: {acc:.6f}")
    pcs, pretty = per_class_accuracy(labels, preds)
    print(f"Round {rd} per-class accuracy -> {pretty}")
    pretty_support, _ = counts_per_class(labels, n_classes=6)
    print(f"Round {rd} test support -> {pretty_support}")

    print(f"Round {rd} time -> query: {_mins(query_sec)} | train: {_mins(train_sec)} | "
          f"test: {_mins(test_sec)} | total: {_mins(total_sec)}")

    # with open(times_csv, "a", newline="") as f:
    #     csv.writer(f).writerow([rd, f"{query_sec:.3f}", f"{train_sec:.3f}",
    #                             f"{test_sec:.3f}", f"{total_sec:.3f}"])
    
    # time row write to csv
    _write_csv_row(time_csv, _time_hdr,
                [rd, f"{query_sec:.3f}", f"{train_sec:.3f}", f"{test_sec:.3f}", f"{total_sec:.3f}"])

    # accuracy row write to csv
    _write_csv_row(acc_csv, _acc_hdr,
                [rd, f"{acc:.6f}"] + [f"{x:.6f}" if x==x else "NA" for x in pcs])

    # log accuracy per round
    to_write = f"strategy_name {args.strategy_name} Round {rd} Acc {acc:.6f} " \
               f"Cls_count {cls_count} Wsi_count {wsi}\n"
    with open(train_log_filepath, "a") as f:
        f.write(to_write)


print("acc_count:{}".format(acc_count))
print('cls_table:{}'.format(cls_table))
print('wsi_count:{}'.format(wsi_name))


# print and log hyperparameters
k_cluster, density_threshold, num_rank_sample = strategy.get_hparams()
# two-blank-column spacer
SPACER = ["", ""]

# time csv: 5 per-round columns, add 2 blanks, 3 hparams
_write_csv_row(time_csv, _time_hdr, ["", "", "", "", ""] + SPACER +
               ["k_cluster", "density_threshold", "num_ranking_sample"])
_write_csv_row(time_csv, _time_hdr, ["", "", "", "", ""] + SPACER +
               [k_cluster, density_threshold, num_rank_sample])

# acc csv: 8 per-round columns (round, avg, I..VI), add 2 blanks, 3 hparams
_write_csv_row(acc_csv, _acc_hdr, ["", "", "", "", "", "", "", ""] + SPACER +
               ["k_cluster", "density_threshold", "num_ranking_sample"])
_write_csv_row(acc_csv, _acc_hdr, ["", "", "", "", "", "", "", ""] + SPACER +
               [k_cluster, density_threshold, num_rank_sample])
print(f"hparams -> k_cluster:{k_cluster} density_threshold:{density_threshold} "
      f"num_ranking_sample:{num_rank_sample}")

to_write = "strategy_name {} Acc {} cls_table {} wsi_count {}\n".format(args.strategy_name, acc_count, cls_table, wsi_name)

with open(train_log_filepath, "a") as f:
    f.write(to_write)
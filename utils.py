import inspect
import torch, inspect, time
from torchvision import transforms
from dataloader import get_data, basic_pool
from nets import Net
from collections import OrderedDict
from ResNet import ResNet101, Res_rank, LossNet, ResNet50, ResNet152
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool, density_cluster, learn_for_loss

params = {
          'MY':
              {'n_epoch': 50,
               'train_args': {'batch_size': 32, 'num_workers': 1},
               'test_args': {'batch_size': 32, 'num_workers': 1},
               'optimizer_args': {'lr': 0.005, 'momentum': 0.9}}

          }


def get_handler(name):
    if name == 'MY' or name == 'WSI':
        return basic_pool


def get_dataset(name):
    if name == 'MY':
        return get_data(get_handler(name))
    elif name == 'WSI':
        return wsi_img(get_handler(name))
    else:
        raise NotImplementedError


def get_net(name, device, root):
    if name == 'MY':
        return Net(ResNet152, params[name], device, root, Res_rank)
    else:
        raise NotImplementedError


def get_params(name):
    return params[name]

def _now():
    # sync GPU so timing is accurate
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def _mins(sec):  # pretty print
    return f"{sec/60:.2f}m"

def _find_inner_module(obj, max_depth=3, _visited=None, _path="root"):
    """Recursively find the first attribute that is a torch.nn.Module (has state_dict)."""
    if _visited is None: _visited = set()
    if id(obj) in _visited or max_depth < 0:
        return None, None
    _visited.add(id(obj))

    if isinstance(obj, torch.nn.Module) or hasattr(obj, "state_dict"):
        return obj, _path

    # Walk attributes; skip primitives and functions/methods
    if hasattr(obj, "__dict__"):
        for name, val in vars(obj).items():
            if isinstance(val, (str, int, float, bool, bytes)):
                continue
            if inspect.isfunction(val) or inspect.ismethod(val):
                continue
            mod, path = _find_inner_module(val, max_depth-1, _visited, f"{_path}.{name}")
            if mod is not None:
                return mod, path
    return None, None

def _load_state_dict_safely(module, ckpt_path, ignore_head=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)  # supports plain sd or dict with 'state_dict'

    # Strip common prefixes
    def strip(k):
        for pref in ("module.", "model.", "net.", "backbone.", "encoder."):
            if k.startswith(pref):
                return k[len(pref):]
        return k

    msd = module.state_dict()
    new_sd = OrderedDict()
    matched, skipped = 0, 0
    for k, v in state.items():
        k2 = strip(k)
        if ignore_head and (
            k2.startswith("fc") or k2.startswith("classifier") or "head" in k2.split(".")
        ):
            skipped += 1
            continue
        if k2 in msd and msd[k2].shape == v.shape:
            new_sd[k2] = v
            matched += 1
        else:
            skipped += 1

    missing, unexpected = module.load_state_dict(new_sd, strict=False)
    print(f"[init] loaded {matched} tensors from {ckpt_path} into {type(module).__name__}")
    if missing:    print("[init] missing keys:", missing)
    if unexpected: print("[init] unexpected keys:", unexpected)
    if skipped:    print(f"[init] skipped {skipped} keys (shape/prefix/head).")

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    elif name == 'MY':
        return density_cluster
    elif name == 'learn_for_loss':
        return learn_for_loss
    else:
        raise NotImplementedError

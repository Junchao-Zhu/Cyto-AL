from torchvision import transforms
from dataloader import get_data, basic_pool
from nets import Net
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

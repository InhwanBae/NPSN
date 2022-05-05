import torch
from torch.utils.data.dataloader import DataLoader
from .model import TrajectoryModel as SGCN
from .utils import TrajectoryDataset


def get_dataloader(data_dir, phase, obs_len, pred_len, batch_size):
    assert phase in ['train', 'val', 'test']
    data_set = data_dir + phase + '/'
    shuffle = True if phase == 'train' else False

    dset_train = TrajectoryDataset(data_set, obs_len=obs_len, pred_len=pred_len, skip=1)
    loader_phase = DataLoader(dset_train, batch_size=1, shuffle=shuffle, num_workers=0)
    return loader_phase, batch_size


def get_latent_dim():
    return 2


def get_model():
    model = SGCN(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5, out_dims=5)
    return model


def model_forward_pre_hook(batch_data, data_sampler=None):
    if data_sampler is not None:
        V_obs, V_tr, _, _ = data_sampler(*[tensor.cuda() for tensor in batch_data[-2:]])
    else:
        V_obs, V_tr = [tensor.cuda() for tensor in batch_data[-2:]]
    return


def model_forward():
    pass


def model_forward_post_hook():
    pass


def model_loss():
    pass

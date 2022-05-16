import torch
from torch.utils.data.dataloader import DataLoader
from .model import social_stgcnn as STGCNN
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
    model = STGCNN(n_stgcnn=1, n_txpcnn=5, output_feat=5, kernel_size=3, seq_len=8, pred_seq_len=12)
    return model


def model_forward_pre_hook(batch_data, data_sampler=None):
    if data_sampler is not None:
        V_obs, V_tr, A_obs, A_tr = data_sampler(*[batch_data[idx].cuda() for idx in [-4, -2, -3, -1]])
    else:
        V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch_data[-4:]]
    return V_obs, A_obs, V_tr, A_tr


def model_forward():
    pass


def model_forward_post_hook():
    pass


def model_loss():
    pass

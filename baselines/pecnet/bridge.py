import yaml
import torch
from torch.utils.data.dataloader import DataLoader
from .model import PECNet as PECNet
from .utils import TrajectoryDataset, TrajBatchSampler, traj_collate_fn


def get_dataloader(data_dir, phase, obs_len, pred_len, batch_size):
    assert phase in ['train', 'val', 'test']
    data_set = data_dir + phase + '/'
    shuffle = True if phase == 'train' else False
    drop_last = True if phase == 'train' else False

    dset_train = TrajectoryDataset(data_set, obs_len=obs_len, pred_len=pred_len)
    sam_phase = TrajBatchSampler(dset_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    loader_phase = DataLoader(dset_train, collate_fn=traj_collate_fn, batch_sampler=sam_phase, pin_memory=True)
    batch_size = 1
    return loader_phase, batch_size


def get_latent_dim():
    return 16


data_scale = 170  # hyper_params["data_scale"]


def get_hyperparams():
    global data_scale
    with open("./baselines/pecnet/optimal_ethucy.yaml", 'r') as file:
        hyper_params = yaml.load(file, Loader=yaml.FullLoader)
        data_scale = hyper_params["data_scale"]
        return (hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"],
                hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'],
                hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"],
                hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'],
                hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], False)


def get_model():
    model = PECNet(*get_hyperparams())
    return model


def model_forward_pre_hook(batch_data, data_sampler=None):
    obs_traj, pred_traj, _, _, mask, _ = [data.cuda(non_blocking=True) for data in batch_data]
    if data_sampler is not None:
        obs_, pred_, _, _ = data_sampler(obs_traj.permute(2, 0, 1).unsqueeze(dim=0),
                                         pred_traj.permute(2, 0, 1).unsqueeze(dim=0))
        obs_traj, pred_traj = obs_.squeeze(dim=0).permute(1, 2, 0), pred_.squeeze(dim=0).permute(1, 2, 0)


def model_forward():
    pass


def model_forward_post_hook():
    pass


def model_loss():
    pass

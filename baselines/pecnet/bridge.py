import yaml
import torch
import numpy as np
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
device = torch.device('cuda')


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

    x = obs_traj.permute(0, 2, 1).clone()
    y = pred_traj.permute(0, 2, 1).clone()

    # starting pos is end of past, start of future. scaled down.
    initial_pos = x[:, 7, :].clone() / 1000

    # shift origin and scale data
    origin = x[:, :1, :].clone()
    x -= origin
    y -= origin
    x *= data_scale  # hyper_params["data_scale"]

    # reshape the data
    x = x.reshape(-1, x.shape[1] * x.shape[2])
    x = x.to(device)
    return obs_traj, pred_traj, mask, x, y, initial_pos, data_scale


def model_forward(model, x, initial_pos, loc):
    all_guesses = []
    for n in range(loc.size(0)):
        dest_recon = model.forward(x, initial_pos, device=device, noise=loc[n])
        all_guesses.append(dest_recon)
    all_guesses = torch.stack(all_guesses, dim=0) / data_scale  # hyper_params["data_scale"]
    return all_guesses


def model_forward_post_hook(model, all_dest_recon, mask, x, y, initial_pos, dest, evaluate_tcc):
    all_guesses = []
    all_l2_errors_dest = []
    for dest_recon in all_dest_recon:
        dest_recon = dest_recon.cpu().numpy()
        all_guesses.append(dest_recon)

        l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
        all_l2_errors_dest.append(l2error_sample)

    all_l2_errors_dest = np.array(all_l2_errors_dest)
    all_guesses = np.array(all_guesses)

    # choosing the best guess
    indices = np.argmin(all_l2_errors_dest, axis=0)
    best_guess_dest = all_guesses[indices, np.arange(x.shape[0]), :]
    best_guess_dest = torch.FloatTensor(best_guess_dest).to(device)

    # using the best guess for interpolation
    interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
    interpolated_future = interpolated_future.cpu().numpy()
    best_guess_dest = best_guess_dest.cpu().numpy()

    # final overall prediction
    predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis=1)
    predicted_future = np.reshape(predicted_future, (-1, 12, 2))  # making sure

    tcc = evaluate_tcc(predicted_future / data_scale, y / data_scale)
    ADEs = np.mean(np.linalg.norm(y - predicted_future, axis=2), axis=1) / data_scale
    FDEs = np.min(all_l2_errors_dest, axis=0) / data_scale
    TCCs = tcc.detach().cpu().numpy()
    return ADEs, FDEs, TCCs


def model_loss(all_guesses, y, loc):
    loss_dist = (all_guesses - y[:, -1].unsqueeze(dim=0)).norm(p=2, dim=-1).min(dim=0)[0].mean()
    loss_disc = (loc.unsqueeze(dim=0) - loc.unsqueeze(dim=1)).norm(p=2, dim=-1)
    loss_disc = loss_disc.topk(k=2, dim=0, largest=False, sorted=True)[0][1].log().mul(-1).mean()
    return loss_dist, loss_disc

import os
import random
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from models.pecnet import PECNet
from models.npsn import NPSN
from models.utils import SocialDataset, box_muller_transform

# Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=512, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=128, help='number of epochs')
parser.add_argument('--num_samples', type=int, default=20, help='number of samples for npsn')
parser.add_argument('--clip_grad', type=float, default=1, help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=32, help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True, help='Use lr rate scheduler')
parser.add_argument('--tag', default='npsn', help='personal tag for the model ')
parser.add_argument('--gpu_num', default="0", type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
torch.set_default_dtype(torch.float64)
device = torch.device('cuda')

metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}


def train(epoch, model, model_npsn, optimizer_npsn, train_dataset, hyper_params):
    global metrics, constant_metrics
    model_npsn.train()
    loss_batch = 0.
    loader_len = len(train_dataset.trajectory_batches)

    torch.autograd.set_detect_anomaly(True)
    for i, (traj, mask, initial_pos) in enumerate(tqdm(zip(train_dataset.trajectory_batches, train_dataset.mask_batches, train_dataset.initial_pos_batches), desc='Train Epoch: {}'.format(epoch), mininterval=1, total=loader_len)):
        optimizer_npsn.zero_grad()

        traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
        x = traj[:, :hyper_params['past_length'], :]
        y = traj[:, hyper_params['past_length']:, :]

        # Augmentation
        scale = random.uniform(0.8, 1.2)
        x *= scale
        y *= scale
        if random.uniform(0, 1) > 0.5:
            x = x[:, :, [1, 0]]
            y = y[:, :, [1, 0]]

        obs_traj = x.unsqueeze(dim=0) / hyper_params["data_scale"]
        pred_traj = y.unsqueeze(dim=0) / hyper_params["data_scale"]

        # NPSN
        loc = model_npsn(obs_traj, mask=mask)
        loc = loc.squeeze(dim=0).permute(1, 0, 2)
        loc = box_muller_transform(loc)

        # reshape the data
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = x.to(device)

        all_guesses = []
        for n in range(args.num_samples):
            dest_recon = model.forward(x, initial_pos, device=device, noise=loc[n])
            all_guesses.append(dest_recon)

        all_guesses = torch.stack(all_guesses, dim=0) / hyper_params["data_scale"]

        loss_dist = (all_guesses - pred_traj.squeeze(dim=0)[:, -1, :].unsqueeze(dim=0)).norm(p=2, dim=-1).min(dim=0)[0].mean()
        loss_disc = (loc.unsqueeze(dim=0) - loc.unsqueeze(dim=1)).norm(p=2, dim=-1)
        loss_disc = loss_disc.topk(k=2, dim=0, largest=False, sorted=True)[0][1].log().mul(-1).mean()

        loss = loss_dist + loss_disc * 0.01
        loss_batch += loss.item()

        loss.backward()
        loss_batch += loss.item()

        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model_npsn.parameters(), args.clip_grad)
        optimizer_npsn.step()

    metrics['train_loss'].append(loss_batch / loader_len)


@torch.no_grad()
def valid(epoch, model, model_npsn, checkpoint_dir, val_dataset, hyper_params):
    global metrics, constant_metrics
    model_npsn.eval()
    loss_batch = 0.
    loader_len = 0

    for i, (traj, mask, initial_pos) in enumerate(tqdm(zip(val_dataset.trajectory_batches, val_dataset.mask_batches, val_dataset.initial_pos_batches), desc='Val Epoch: {}'.format(epoch), mininterval=1, total=len(val_dataset.trajectory_batches))):
        traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
        x = traj[:, :hyper_params['past_length'], :]
        y = traj[:, hyper_params['past_length']:, :]

        obs_traj = x.unsqueeze(dim=0) / hyper_params["data_scale"]
        pred_traj = y.unsqueeze(dim=0) / hyper_params["data_scale"]

        # NPSN
        loc = model_npsn(obs_traj, mask=mask)
        loc = loc.squeeze(dim=0).permute(1, 0, 2)
        loc = box_muller_transform(loc)

        # reshape the data
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = x.to(device)

        all_guesses = []
        for n in range(args.num_samples):
            dest_recon = model.forward(x, initial_pos, device=device, noise=loc[n])
            all_guesses.append(dest_recon)

        all_guesses = torch.stack(all_guesses, dim=0) / hyper_params["data_scale"]

        loss_dist = (all_guesses - pred_traj.squeeze(dim=0)[:, -1, :].unsqueeze(dim=0)).norm(p=2, dim=-1).min(dim=0)[0].sum()
        loss = loss_dist
        loss_batch += loss.item()
        loader_len += loc.size(1)

    metrics['val_loss'].append(loss_batch / loader_len)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model_npsn.state_dict(), checkpoint_dir + 'val_best.pth')


def main(args):
    print("Training initiating....")
    checkpoint_dir = './checkpoints/' + args.tag + '/'
    checkpoint = torch.load('./saved_models/PECNET_social_model1.pt', map_location=device)
    hyper_params = checkpoint["hyper_params"]

    model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"],
                   hyper_params["dec_size"], hyper_params["predictor_hidden_size"],
                   hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'],
                   hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"],
                   hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"],
                   hyper_params["past_length"], hyper_params["future_length"], False)
    model = model.double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_dataset = SocialDataset(set_name="train", b_size=hyper_params["train_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=False)
    test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=False)

    # shift origin and scale data
    for traj in train_dataset.trajectory_batches:
        traj -= traj[:, :1, :]
        traj *= hyper_params["data_scale"]
    for traj in test_dataset.trajectory_batches:
        traj -= traj[:, :1, :]
        traj *= hyper_params["data_scale"]

    # load baseline model and npsn
    model_npsn = NPSN(t_obs=args.obs_len, s=16, n=args.num_samples).cuda()
    optimizer_npsn = torch.optim.AdamW(model_npsn.parameters(), lr=args.lr)

    if args.use_lrschd:
        scheduler_npsn = torch.optim.lr_scheduler.StepLR(optimizer_npsn, step_size=args.lr_sh_rate, gamma=0.5)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    for epoch in range(args.num_epochs):
        train(epoch, model, model_npsn, optimizer_npsn, train_dataset, hyper_params)
        valid(epoch, model, model_npsn, checkpoint_dir, test_dataset, hyper_params)

        if args.use_lrschd:
            scheduler_npsn.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format('SDD', epoch))
        print("Train_loss: {0:.8f}, Val_los: {1:.8f}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1:.8f}".format(constant_metrics['min_val_epoch'],
                                                                 constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

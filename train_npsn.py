import os
import random
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from baselines.converter import get_identity
from npsn import *

# Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='zara2', help='scene ["eth","hotel","univ","zara1","zara2"]')
parser.add_argument('--baseline', default='sgcn', help='baseline network ["sgcn","stgcnn"]')
parser.add_argument('--batch_size', type=int, default=512, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=128, help='number of epochs')
parser.add_argument('--num_samples', type=int, default=20, help='number of samples for npsn')
parser.add_argument('--clip_grad', type=float, default=1, help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=32, help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True, help='Use lr rate scheduler')
parser.add_argument('--tag', default='npsn', help='personal tag for the model ')
parser.add_argument('--gpu_num', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}

if args.baseline == 'stgcnn':
    from baselines.stgcnn import *
elif args.baseline == 'sgcn':
    from baselines.sgcn import *


def train(epoch, model, model_npsn, optimizer_npsn, loader_train):
    global metrics, constant_metrics
    model_npsn.train()
    loss_batch = 0.
    loader_len = len(loader_train)

    for cnt, batch in enumerate(tqdm(loader_train, desc='Train Epoch: {}'.format(epoch), mininterval=1)):
        if cnt % args.batch_size == 0:
            optimizer_npsn.zero_grad()

        if args.baseline == 'stgcnn':
            V_obs, V_tr, A_obs, A_tr = data_sampler(*[tensor.cuda() for tensor in batch[[-4, -2, -3, -1]]])
            with torch.no_grad():
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)
                V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
                V_pred = V_pred.permute(0, 2, 3, 1).detach()
        elif args.baseline == 'sgcn':
            V_obs, V_tr, _, _ = data_sampler(*[tensor.cuda() for tensor in batch[-2:]])
            identity = get_identity(V_obs.shape)
            with torch.no_grad():
                V_pred = model(V_obs, identity).detach()
            V_obs = V_obs[..., 1:]

        mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
        loc = model_npsn(V_obs.permute(0, 2, 3, 1))
        loss_dist, loss_disc = model_npsn.get_loss(loc, mu, cov, V_tr.permute(0, 2, 3, 1))
        loss = loss_dist * 1.0 + loss_disc * 0.01
        loss.backward()
        loss_batch += loss.item()

        if cnt % args.batch_size + 1 == args.batch_size:  # or cnt + 1 == loader_len:  # drop last
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model_npsn.parameters(), args.clip_grad)
            optimizer_npsn.step()

    metrics['train_loss'].append(loss_batch / loader_len)


@torch.no_grad()
def valid(epoch, model, model_npsn, checkpoint_dir, loader_val):
    global metrics, constant_metrics
    model_npsn.eval()
    loss_batch = 0.
    loader_len = 0

    for cnt, batch in enumerate(tqdm(loader_val, desc='Valid Epoch: {}'.format(epoch), mininterval=1)):
        obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

        if args.baseline == 'stgcnn':
            V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
            V_pred = V_pred.permute(0, 2, 3, 1).detach()
        elif args.baseline == 'sgcn':
            V_obs, V_tr = [tensor.cuda() for tensor in batch[-2:]]
            identity = get_identity(V_obs.shape)
            V_pred = model(V_obs, identity)
            V_obs = V_obs[..., 1:]

        mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
        loc = model_npsn(V_obs.permute(0, 2, 3, 1))

        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

        # Randomly sampling predict trajectories
        V_pred_sample = purposive_sample(mu, cov, loc.size(2), loc)

        # Evaluate trajectories
        V_absl = V_pred_sample.cumsum(dim=1) + V_obs_traj[[-1], :, :]
        ADEs, FDEs, TCCs = compute_batch_metric(V_absl, V_pred_traj_gt)

        loss_batch += FDEs.sum().item()
        loader_len += FDEs.size(0)

    metrics['val_loss'].append(loss_batch / loader_len)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model_npsn.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args):
    print("Training initiating....")
    print(args)

    data_set = './dataset/' + args.dataset + '/'
    model_path = './pretrained/' + args.baseline + '/' + args.dataset + '/val_best.pth'
    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'

    dset_train = TrajectoryDataset(data_set + 'train/', obs_len=args.obs_len, pred_len=args.pred_len, skip=1)
    loader_train = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=0)
    dset_val = TrajectoryDataset(data_set + 'val/', obs_len=args.obs_len, pred_len=args.pred_len, skip=1)
    loader_val = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=0)

    # Load baseline network model and npsn
    if args.baseline == 'stgcnn':
        model = STGCNN(n_stgcnn=1, n_txpcnn=5, output_feat=5, kernel_size=3, seq_len=8, pred_seq_len=12).cuda()
    elif args.baseline == 'sgcn':
        model = SGCN(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                     obs_len=8, pred_len=12, n_tcn=5, out_dims=5).cuda()
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model_npsn = NPSN(t_obs=args.obs_len, s=2, n=args.num_samples).cuda()
    print('{} parameters:'.format(args.baseline), count_parameters(model))
    print('npsn parameters:', count_parameters(model_npsn))

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
        train(epoch, model, model_npsn, optimizer_npsn, loader_train)
        valid(epoch, model, model_npsn, checkpoint_dir, loader_val)

        if args.use_lrschd:
            scheduler_npsn.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.dataset, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(constant_metrics['min_val_epoch'],
                                                             constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

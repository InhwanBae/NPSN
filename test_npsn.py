import os
import glob
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from baselines.converter import get_sgcn_identity
from npsn import *

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default='sgcn', help='baseline network ["sgcn","stgcnn","pecnet"]')
parser.add_argument('--method', default='npsn', help='sampling method ["mc","qmc","npsn"]')
parser.add_argument('--tag', default='npsn', help='personal tag for the model')
parser.add_argument('--gpu_num', default='0', type=str)

test_args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = test_args.gpu_num

if test_args.baseline == 'stgcnn':
    from baselines.stgcnn import *
elif test_args.baseline == 'sgcn':
    from baselines.sgcn import *
elif test_args.baseline == 'pecnet':
    from baselines.pecnet import *
else:
    raise NotImplementedError


@torch.no_grad()
def test(model, model_npsn, loader_test, method='npsn', samples=20, trials=100):
    model.eval()
    model_npsn.eval()
    ade_all, fde_all, tcc_all = [], [], []

    if method == 'qmc':
        sobol_generator = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=0)

    for batch in tqdm(loader_test, desc=loader_test.dataset.data_dir):
        if test_args.baseline in ['stgcnn', 'sgcn']:
            obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

            if test_args.baseline == 'stgcnn':
                V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)
                V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
                V_pred = V_pred.permute(0, 2, 3, 1)
            elif test_args.baseline == 'sgcn':
                V_obs, V_tr = [tensor.cuda() for tensor in batch[-2:]]
                identity = get_sgcn_identity(V_obs.shape)
                V_pred = model(V_obs, identity)
                V_obs = V_obs[..., 1:]

            mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))

            if method == 'npsn':
                loc = model_npsn(V_obs.permute(0, 2, 3, 1))

            V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
            V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

            ade_stack, fde_stack, tcc_stack = [], [], []
            for trial in range(trials):
                if method == 'mc':
                    V_pred_sample = mc_sample(mu, cov, samples)
                elif method == 'qmc':
                    V_pred_sample = qmc_sample(mu, cov, samples, sobol_generator)
                elif method == 'npsn':
                    V_pred_sample = purposive_sample(mu, cov, samples, loc)
                else:
                    raise NotImplementedError

                # Evaluate trajectories
                V_absl = V_pred_sample.cumsum(dim=1) + V_obs_traj[[-1], :, :]
                ADEs, FDEs, TCCs = compute_batch_metric(V_absl, V_pred_traj_gt)

                ade_stack.append(ADEs.detach().cpu().numpy())
                fde_stack.append(FDEs.detach().cpu().numpy())
                tcc_stack.append(TCCs.detach().cpu().numpy())

        elif test_args.baseline == 'pecnet':
            obs_traj, pred_traj, mask, x, y, initial_pos, data_scale = model_forward_pre_hook(batch)
            device = torch.device('cuda')
            y *= data_scale  # hyper_params["data_scale"]
            y = y.cpu().numpy()
            dest = y[:, -1, :]

            ade_stack, fde_stack, tcc_stack = [], [], []
            for trial in range(trials):
                # NPSN
                if method == 'qmc':
                    sobol_generator = torch.quasirandom.SobolEngine(dimension=16, scramble=True)
                    loc = box_muller_transform(sobol_generator.draw(samples).cuda()).unsqueeze(dim=1).expand((samples, x.size(0), 16))
                elif method == 'npsn':
                    loc = model_npsn(obs_traj.unsqueeze(dim=0).transpose(-1, -2), mask=mask)
                    loc = loc.squeeze(dim=0).permute(1, 0, 2)
                    loc = box_muller_transform(loc)

                all_dest_recon = []
                for n in range(samples):
                    if method == 'mc':
                        dest_recon = model.forward(x, initial_pos, device=device)
                    elif method == 'qmc':
                        dest_recon = model.forward(x, initial_pos, device=device, noise=loc[n])
                    elif method == 'npsn':
                        dest_recon = model.forward(x, initial_pos, device=device, noise=loc[n])
                    else:
                        raise NotImplementedError

                    all_dest_recon.append(dest_recon)

                ADEs, FDEs, TCCs = model_forward_post_hook(model, all_dest_recon, mask, x, y, initial_pos, dest, evaluate_tcc)

                ade_stack.append(ADEs)
                fde_stack.append(FDEs)
                tcc_stack.append(TCCs)

        ade_all.append(np.array(ade_stack))
        fde_all.append(np.array(fde_stack))
        tcc_all.append(np.array(tcc_stack))

    ade_all = np.concatenate(ade_all, axis=1)
    fde_all = np.concatenate(fde_all, axis=1)
    tcc_all = np.concatenate(tcc_all, axis=1)

    mean_ade, mean_fde, mean_tcc = ade_all.mean(axis=0).mean(), fde_all.mean(axis=0).mean(), tcc_all.mean(axis=0).mean()
    return mean_ade, mean_fde, mean_tcc


def main():
    ADE_ls, FDE_ls, TCC_ls = [], [], []
    print("*" * 50)
    root_ = './checkpoints/' + test_args.tag + '-' + test_args.baseline + '/'
    dataset = ['eth', 'hotel', 'univ', 'zara1', 'zara2']

    paths = list(map(lambda x: root_ + x, dataset))

    for feta in range(len(paths)):
        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:', exps)
        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            data_set = './dataset/' + args.dataset + '/'
            model_path = './pretrained/' + test_args.baseline + '/' + args.dataset + '/val_best.pth'
            model_npsn_path = exp_path + '/val_best.pth'

            # Dataloader
            loader_test, _ = get_dataloader(data_set, 'test', args.obs_len, args.pred_len, args.batch_size)

            # Load backbone network and NPSN
            model = get_model().cuda()
            model.load_state_dict(torch.load(model_path))
            model_npsn = NPSN(t_obs=args.obs_len, s=get_latent_dim(), n=args.num_samples).cuda()
            model_npsn.load_state_dict(torch.load(model_npsn_path))

            ADE, FDE, TCC = test(model, model_npsn, loader_test, test_args.method.lower(), args.num_samples)
            ADE_ls.append(ADE), FDE_ls.append(FDE), TCC_ls.append(TCC)
            print("Method: {} N: {} ADE: {:.8f} FDE: {:.8f} TCC: {:.8f}".format(test_args.method.upper(),
                                                                                args.num_samples, ADE, FDE, TCC))

    print("*" * 50)
    print("AVG ADE: {:.8f} AVG FDE: {:.8f} AVG TCC: {:.8f}".format(sum(ADE_ls) / 5, sum(FDE_ls) / 5, sum(TCC_ls) / 5))
    print("*" * 50)


if __name__ == '__main__':
    main()

import os
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from models.pecnet import PECNet
from models.npsn import NPSN
from models.utils import SocialDataset, box_muller_transform, evaluate_tcc

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='npsn', help='personal tag for the model ')
parser.add_argument('--method', default='npsn', help='sampling method ["mc","qmc","npsn"]')
parser.add_argument('--gpu_num', default="0", type=str)

test_args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = test_args.gpu_num
torch.set_default_dtype(torch.float64)
device = torch.device('cuda')


@torch.no_grad()
def test(model, model_npsn, test_dataset, hyper_params={}, method='npsn', samples=20, trials=100):
    model.eval()
    model_npsn.eval()
    ade_all, fde_all, tcc_all = [], [], []

    for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
        traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
        x = traj[:, :hyper_params['past_length'], :]
        y = traj[:, hyper_params['past_length']:, :]
        y = y.cpu().numpy()

        obs_traj = x.unsqueeze(dim=0) / hyper_params["data_scale"]

        # reshape the data
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = x.to(device)
        dest = y[:, -1, :]

        ade_stack, fde_stack, tcc_stack = [], [], []

        for trial in tqdm(range(trials)):
            if method == 'qmc':
                sobol_generator = torch.quasirandom.SobolEngine(dimension=16, scramble=True)
                noise_sobol = box_muller_transform(sobol_generator.draw(samples).cuda()).unsqueeze(dim=1).expand((samples, x.size(0), 16))
            elif method == 'npsn':
                loc = model_npsn(obs_traj, mask=mask)
                loc = loc.squeeze(dim=0).permute(1, 0, 2)
                loc = box_muller_transform(loc)

            all_guesses = []
            all_l2_errors_dest = []

            for n in range(samples):
                if method == 'mc':
                    dest_recon = model.forward(x, initial_pos, device=device)
                elif method == 'qmc':
                    dest_recon = model.forward(x, initial_pos, device=device, noise=noise_sobol[n])
                elif method == 'npsn':
                    dest_recon = model.forward(x, initial_pos, device=device, noise=loc[n])
                else:
                    raise NotImplementedError

                dest_recon = dest_recon.cpu().numpy()
                all_guesses.append(dest_recon)

                l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
                all_l2_errors_dest.append(l2error_sample)

            all_l2_errors_dest = np.array(all_l2_errors_dest)
            all_guesses = np.array(all_guesses)

            # choosing the best guess
            indices = np.argmin(all_l2_errors_dest, axis=0)

            best_guess_dest = all_guesses[indices, np.arange(x.shape[0]), :]
            best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

            # using the best guess for interpolation
            interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
            interpolated_future = interpolated_future.cpu().numpy()
            best_guess_dest = best_guess_dest.cpu().numpy()

            # final overall prediction
            predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis=1)
            predicted_future = np.reshape(predicted_future, (-1, hyper_params['future_length'], 2))  # making sure

            tcc = evaluate_tcc(predicted_future / hyper_params["data_scale"], y / hyper_params["data_scale"])

            ADEs = np.mean(np.linalg.norm(y - predicted_future, axis=2), axis=1)
            FDEs = np.min(all_l2_errors_dest, axis=0)
            TCCs = tcc.detach().cpu().numpy()

            ade_stack.append(ADEs / hyper_params["data_scale"])
            fde_stack.append(FDEs / hyper_params["data_scale"])
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
    exp_path = './checkpoints/' + test_args.tag + '/'
    print("Evaluating model:", exp_path)

    args_path = exp_path + '/args.pkl'
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
        model_npsn_path = exp_path + '/val_best.pth'
        checkpoint = torch.load('./saved_models/PECNET_social_model1.pt', map_location=device)
        hyper_params = checkpoint["hyper_params"]

        model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], False)
        model = model.double().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=False)

        for traj in test_dataset.trajectory_batches:
            traj -= traj[:, :1, :]
            traj *= hyper_params["data_scale"]

        model_npsn = NPSN(t_obs=args.obs_len, s=16, n=args.num_samples).cuda()
        model_npsn.load_state_dict(torch.load(model_npsn_path))

        # Evaluation
        ADE, FDE, TCC = test(model, model_npsn, test_dataset, hyper_params, method=test_args.method.lower(), samples=args.num_samples)
        print("Method: {} N: {} ADE: {:.8f} FDE: {:.8f} TCC: {:.8f}".format(test_args.method, args.num_samples, ADE, FDE, TCC))


if __name__ == '__main__':
    main()

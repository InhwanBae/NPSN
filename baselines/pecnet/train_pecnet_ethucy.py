import yaml
import pickle
import argparse
from tqdm import tqdm
from utils import *
from model import PECNet


parser = argparse.ArgumentParser(description='PECNet')
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal_ethucy.yaml')
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--dataset', default='zara2', help='scene ["eth","hotel","univ","zara1","zara2"]')
parser.add_argument('--tag', default='pecnet', help='personal tag for the model ')
parser.add_argument('--gpu_num', default='0', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
device = torch.device('cuda')


def train(epoch, model, optimizer, loader, hyper_params):
    model.train()
    train_loss = 0
    total_rcl, total_kld, total_adl = 0, 0, 0
    criterion = torch.nn.MSELoss()

    for cnt, batch in enumerate(tqdm(loader, desc='Train Epoch: {}'.format(epoch), mininterval=1)):
        batch = [data.cuda(non_blocking=True) for data in batch]
        obs_traj, pred_traj, _, _, mask, _ = batch

        # augment training set with reversed tracklets...
        if torch.rand(1).item() > 0.5:
            t = torch.cat([obs_traj, pred_traj], dim=-1)
            reverse_t = torch.flip(t, dims=[-1])
            obs_traj = reverse_t[..., :8]
            pred_traj = reverse_t[..., 8:]

        x = obs_traj.permute(0, 2, 1).clone()
        y = pred_traj.permute(0, 2, 1).clone()

        # starting pos is end of past, start of future. scaled down.
        initial_pos = x[:, 7, :].clone() / 1000

        # shift origin and scale data
        origin = x[:, :1, :].clone()
        x -= origin
        y -= origin
        x *= hyper_params["data_scale"]
        y *= hyper_params["data_scale"]

        # reshape the data
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = x.to(device)
        dest = y[:, -1, :].to(device)
        future = y[:, :-1, :].contiguous().view(y.size(0), -1).to(device)

        dest_recon, mu, var, interpolated_future = model.forward(x, initial_pos, dest=dest, mask=mask, device=device)

        optimizer.zero_grad()
        rcl, kld, adl = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future)
        loss = rcl + kld * hyper_params["kld_reg"] + adl * hyper_params["adl_reg"]
        loss.backward()

        train_loss += loss.item()
        total_rcl += rcl.item()
        total_kld += kld.item()
        total_adl += adl.item()
        optimizer.step()

    return train_loss, total_rcl, total_kld, total_adl


@torch.no_grad()
def valid(epoch, model, loader, hyper_params, best_of_n=1):
    assert best_of_n >= 1 and type(best_of_n) == int

    model.eval()
    total_seen = 0
    total_ade = 0
    total_fde = 0

    for cnt, batch in enumerate(tqdm(loader, desc='Valid Epoch: {}'.format(epoch), mininterval=1)):
        batch = [data.cuda(non_blocking=True) for data in batch]
        obs_traj, pred_traj, _, _, mask, _ = batch

        x = obs_traj.permute(0, 2, 1).clone()
        y = pred_traj.permute(0, 2, 1).clone()

        # starting pos is end of past, start of future. scaled down.
        initial_pos = x[:, 7, :].clone() / 1000

        # shift origin and scale data
        origin = x[:, :1, :].clone()
        x -= origin
        y -= origin
        x *= hyper_params["data_scale"]
        y *= hyper_params["data_scale"]

        total_seen += len(obs_traj)
        y = y.cpu().numpy()

        # reshape the data
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = x.to(device)

        dest = y[:, -1, :]
        all_l2_errors_dest = []
        all_guesses = []
        for _ in range(best_of_n):
            dest_recon = model.forward(x, initial_pos, device=device)
            dest_recon = dest_recon.cpu().numpy()
            all_guesses.append(dest_recon)

            l2error_sample = np.linalg.norm(dest_recon - dest, axis=1)
            all_l2_errors_dest.append(l2error_sample)

        all_l2_errors_dest = np.array(all_l2_errors_dest)
        all_guesses = np.array(all_guesses)

        # average error
        l2error_avg_dest = np.mean(all_l2_errors_dest)

        # choosing the best guess
        indices = np.argmin(all_l2_errors_dest, axis=0)
        best_guess_dest = all_guesses[indices, np.arange(x.shape[0]), :]

        # taking the minimum error out of all guess
        l2error_dest = np.mean(np.min(all_l2_errors_dest, axis=0))
        best_guess_dest = torch.FloatTensor(best_guess_dest).to(device)

        # using the best guess for interpolation
        interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
        interpolated_future = interpolated_future.cpu().numpy()
        best_guess_dest = best_guess_dest.cpu().numpy()

        # final overall prediction
        predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis=1)
        predicted_future = np.reshape(predicted_future, (-1, hyper_params['future_length'], 2))  # making sure
        # ADE error
        l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis=2))

        l2error_overall /= hyper_params["data_scale"]
        l2error_dest /= hyper_params["data_scale"]
        l2error_avg_dest /= hyper_params["data_scale"]

        total_ade += (l2error_overall * len(obs_traj))
        total_fde += (l2error_dest * len(obs_traj))

    return total_ade / total_seen, total_fde / total_seen


def main(hyper_params):
    print("Training initiating....")
    print(hyper_params)

    data_set = '../../dataset/' + args.dataset + '/'
    checkpoint_dir = '../../checkpoints/' + args.tag + '/' + args.dataset + '/'
    obs_len, pred_len = hyper_params["past_length"], hyper_params["future_length"]

    train_dataset = TrajectoryDataset(data_set + 'train/', obs_len=obs_len, pred_len=pred_len)
    train_sampler = TrajBatchSampler(train_dataset, batch_size=hyper_params["train_b_size"], shuffle=True)
    train_loader = DataLoader(train_dataset, collate_fn=traj_collate_fn, batch_sampler=train_sampler, pin_memory=True)

    val_dataset = TrajectoryDataset(data_set + 'val/', obs_len=obs_len, pred_len=pred_len)
    val_sampler = TrajBatchSampler(val_dataset, batch_size=hyper_params["test_b_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, collate_fn=traj_collate_fn, batch_sampler=val_sampler, pin_memory=True)

    model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"],
                   hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'],
                   hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"],
                   hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'],
                   hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])

    best_test_loss = 50  # start saving after this threshold
    best_endpoint_loss = 50
    N = hyper_params["n_values"]

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    for epoch in range(hyper_params['num_epochs']):
        train_loss, rcl, kld, adl = train(epoch, model, optimizer, train_loader, hyper_params)
        test_loss, final_point_loss_best = valid(epoch, model, val_loader, hyper_params, best_of_n=N)

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            save_path = checkpoint_dir + 'val_best.pth'
            torch.save(model.state_dict(), save_path)
            best_endpoint_loss = final_point_loss_best

        print("Epoch: {}, ADE: {:.4f}, FDE: {:.4f}, Min ADE: {:.4f}, Min FDE: {:.4f}".format(epoch, test_loss, final_point_loss_best, best_test_loss, best_endpoint_loss))


if __name__ == '__main__':
    with open("./" + args.config_filename, 'r') as file:
        hyper_params = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    main(hyper_params)

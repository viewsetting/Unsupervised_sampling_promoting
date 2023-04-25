import numpy as np
import torch

def model_forward_post_hook(model, all_dest_recon, mask, x, y, initial_pos, dest,data_scale=170 ):
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
    best_guess_dest = torch.FloatTensor(best_guess_dest).to(x.device)

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
    return ADEs, FDEs, 

def evaluate_tcc(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
    pred, gt = torch.FloatTensor(pred).permute(1, 0, 2), torch.FloatTensor(gt).permute(1, 0, 2)
    pred_best = pred
    pred_gt_stack = torch.stack([pred_best.permute(1, 0, 2), gt.permute(1, 0, 2)], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef.clip_(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return TCCs
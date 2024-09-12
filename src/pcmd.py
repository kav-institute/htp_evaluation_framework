import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from termcolor import colored
from tqdm import tqdm


def calc_pcmd(gt, gmm, k, pcmd_levels, dir, with_plot=False):
    """calculate pcmd scores, batched
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        k (int): number k of samples to get from distributions
        pcmd_levels (list): discrete pcmd sample selection levels
        dir (string): destination data directory
        with_plot (bool, optional): with or without plot
    Returns:
        dict: pcmd socres
    """
    # gt: [n_tracks, n_forecast_horizons, n_features]
    # gmm: [n_tracks, n_forecast_horizons]
    
    print(colored(f" - Calculating pcmd scores", 'green'))
    
    # get samples with probabilities from mixtures
    res_dict = {}
    ade_list = []
    fde_list = []
    samples, probs = build_samples_with_probabilities(gmm=gmm, k=k)
    
    # get results for different number of used samples k
    for eval_step in tqdm(pcmd_levels):
        
        _, top_k_indices = torch.topk(probs, k=eval_step, dim=0)
        top_k_samples = samples.gather(dim=0, index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, 2))
        
        # compute ade/fde
        ade = compute_ade(predictions=top_k_samples, gt=gt)
        fde = compute_fde(predictions=top_k_samples, gt=gt)
        
        # save
        ade_list.append(ade)
        fde_list.append(fde)
        res_dict[f"PCMD: ADE/FDE @ {eval_step}/{k}"] = (ade, fde)
        
    if with_plot:
        
        # create figure
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10.5))
        
        axs[0].set_box_aspect(aspect=1)
        axs[0].grid() 
        axs[0].plot(np.array(pcmd_levels), np.array(ade_list), color='b', linewidth=2, label="PCMD_ADE", marker='.', markersize=12)
        axs[0].set_xlabel("Rank", fontsize = 18)
        axs[0].set_ylabel("ADE in (m)", fontsize = 18)
        
        axs[1].set_box_aspect(aspect=1)
        axs[1].grid() 
        axs[1].plot(np.array(pcmd_levels), np.array(fde_list), color='r', linewidth=2, label="PCMD_FDE", marker='.', markersize=12)
        axs[1].set_xlabel("Rank", fontsize = 18)
        axs[1].set_ylabel("FDE in (m)", fontsize = 18)
        
        fig.tight_layout()
        plt.suptitle(f'PCMD Scores for Top K Ranking: K={pcmd_levels} @ {k} Samples', fontsize=20)
        plt.savefig(os.path.join(dir, f'pcmd_curves.png'))
        plt.close()
    
    return res_dict


def compute_ade(predictions, gt):
    """ batched min ade calculation
    Args:
        predictions (torch.tensor): k predicted samples [batch_size, n_horizons, k, n_features]
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
    Returns:
        float: min ade result
    """
    
    error = torch.linalg.norm(predictions - gt, axis=-1)
    ade = torch.mean(error, axis=-1)
    min_ade = torch.min(ade, dim=0).values.mean().item()
    return min_ade


def compute_fde(predictions, gt):
    """ batched min fde calculation
    Args:
        predictions (torch.tensor): k predicted samples [batch_size, n_horizons, k, n_features]
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
    Returns:
        float: min fde result
    """
    
    fde = torch.linalg.norm(predictions - gt, axis=-1)[:,:,-1]
    min_fde = torch.min(fde, dim=0).values.mean().item()
    return min_fde


def build_samples_with_probabilities(gmm, k):
    """get samples from distribution with probabilites derived from density
    Args:
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        k (int): number of samples to get from each distribution
    Returns:
        torch.tensor: samples and corresponding probabilities
    """
    
    # get samples and compute log probabilities
    tau = 1
    samples = gmm.sample(sample_shape=torch.Size([k]))
    samples_log_prob = gmm.log_prob(samples)
    probs = torch.exp(samples_log_prob/tau) / torch.sum(torch.exp(samples_log_prob/tau), dim=0)
    
    return samples, probs
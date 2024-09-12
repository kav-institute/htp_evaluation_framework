import os
import sys
import torch
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from termcolor import colored
from skimage import measure

sys.path.append("/workspace/repos/src")


def calc_reliability(gt, gmm, n_sample_points, percentiles, n_forecasts_horizons, dt, dir, with_plot=False):
    """calculate reliability scores and create reliability calibration plot
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        n_sample_points (int): number of samples to get from each distribution
        percentiles(list): list containing reliability bins
        n_forecast_horizons (int): number of discrete forecast horizons
        dt (float): delta t between two discrete timesteps of forecast horizon
        dir (string): destination data directory
        with_plot (bool, optional): with or without reliability calibration plot
    Returns:
        _type_: _description_
    """
    
    print(colored(f" - Calculating realiability scores", 'green'))
    
    # get confidence sets
    res_dict = {}
    confidence_sets = build_confidence_set(gt=gt, gmm=gmm, n_sample_points=n_sample_points)
    
    # place/sort values into bins
    # attention!: digitize() returns indexes, with first index starting at 1 (not 0)
    bin_data = np.digitize(confidence_sets, bins=percentiles)
    reliability_errors = []
    
    if with_plot:
        
        plt.figure()
        plt.plot(percentiles, percentiles, 'k--', linewidth=3, label=f"ideal")
    
    # do for every forecast horizon
    for idx in tqdm(range(0, n_forecasts_horizons)):
        
        # build calibration curve
        # attention!: bincount() returns amount of each bin, first bin to count is bin at 0,
        # due to above digitize behavior, we must increment len(bins) by 1 and later ignore the zero bin count result
        f0 = np.array(np.bincount(bin_data[:,idx], minlength=len(percentiles)+1)).T
        
        # f0[1:]: because of the different start values of digitize and bincount, we remove/ignore the first value of f0
        acc_f0 = np.cumsum(f0[1:],axis=0)/confidence_sets.shape[0]
        
        # get differences for current step
        reliability_errors.append(abs(acc_f0 - percentiles))
        
        if with_plot: plt.plot(percentiles ,acc_f0, linewidth=3, label=f"{round((idx+1)*dt, 1)} sec")
        
    # get reliability scores
    avg_rls = (1 - np.mean(reliability_errors))*100
    min_rls = (1 - np.max(reliability_errors))*100
    res_dict[f"avg_RLS:"] = avg_rls
    res_dict[f"min_RLS:"] = min_rls
    
    if with_plot:
        
        plt.grid()
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title(f'Reliability Calibration Curve')
        plt.legend(fontsize = 8)
        plt.savefig(os.path.join(dir, f'reliability_calibration.png'))
        plt.close()
    
    return res_dict


def calc_sharpness_sets(gmm, samples, n_sample_points, mesh_resolution, confidence_levels):
    """calcluate sharpness sets for each track
    Args:
        gmm (torch.distribution.MixtureSameFamily): single track gaussian mixture model [1, n_horizons]
        samples (torch.tensor): k predicted samples [k, batch_size, n_horizons, n_features]
        n_sample_points (int): number of samples to get from each distribution
        mesh_resolution (int): size of mesh grid
        confidence_levels (list): list object containing a number of user defined confidence levels
    Returns:
        torch.tensor: sharpness sets for each confidence level
    """
    
    grid, _, _, range = build_mesh_grid(samples=samples, mesh_resolution=mesh_resolution)
    confidence_map = build_confidence_set(gt=grid, gmm=gmm, n_sample_points=n_sample_points)
    
    # handle different confidence levels
    sharpness = []
    for k in confidence_levels:
        
        # calc sharpness
        sharpness.append(estimate_sharpness(confidence_map, kappa=k)*(range*range))
        
    s = torch.stack(sharpness, 0)
    
    return s


def calc_sharpness(sharpness_sets, confidence_levels, percentiles, n_forecasts_horizons, dt):
    """calculate sharpness scores from sharpness sets
    Args:
        sharpness_sets (torch.tensor): sharpness sets for each confidence level
        confidence_levels (list): list object containing a number of user defined confidence levels
        percentiles (list): list containing bins
        n_forecast_horizons (int): number of discrete forecast horizon
        dt (float): delta t between two discrete timesteps of forecast horizon
    Returns:
        dict: sharpness scores
    """
    
    sharpness_scores = {}
    
    for idx, cl in enumerate(confidence_levels):
        
        s = sharpness_sets[:,idx,:].T
        SDist=np.zeros((n_forecasts_horizons, len(percentiles)))
        
        for k, p in enumerate(percentiles):
            
            for t in range(n_forecasts_horizons):
                
                SDist[t,k]=np.percentile(s[t,:], p*100, axis=-1)
        
        # calc mean sharpness score (i.e 50%) for current confidence level
        sharpness_score = sum([np.mean(SDist[idx,:] / ((step+1)*dt)) for idx, step in enumerate(range(n_forecasts_horizons))]) * (1/(n_forecasts_horizons*dt))
        sharpness_scores[f"SS @ {cl}"] = sharpness_score
    
    return sharpness_scores


def build_confidence_levels(confidence_map, modes, confidence_levels, n_forecast_horizons, steps, mesh_resolution, r):
    """get discrete confidence areas in world coordinates for each object and forcast horizon
    Args:
        confidence_map (torch.tensor): confidence map of single objects prediction
        modes (torch.tensor): modes of every mixture
        confidence_levels (list): list object containing a number of user defined confidence levels
        n_forecast_horizons (int): number of discrete forecast horizon
        steps (int): number of mesh grid steps
        mesh_resolution (float): resolution of mesh grid
        r (int): mesh grid range
    Returns:
        list: list of contours for each track, timestep and confidence level
    """
    
    confidence_areas = []
    confidence_contours = []
    
    for k in confidence_levels:
        
        confidence_areas.append(torch.reshape(torch.where(confidence_map <= k, 1, 0), (steps, steps, n_forecast_horizons)).numpy()) 
    
    for idx, _ in enumerate(confidence_levels):
        
        contours = []
        
        for h in range(n_forecast_horizons):
            
            # get contour(s) of confidence level and time step
            c = measure.find_contours(confidence_areas[idx][:, :, h], 0.5)
            
            # uni modal dist
            if len(c) == 1:
                
                cont = [np.flip(m=np.squeeze(np.array(c, dtype=np.float32) * mesh_resolution - r))]
                contours.append([np.squeeze(a=cont, axis=0)])
                
            # multi modal dist
            elif len(c) > 1:
                
                cont = [np.array(c[i], dtype=np.float32) * mesh_resolution - r for i in range(0, len(c))]
                cont = [np.flip(m=ct) for ct in cont]
                contours.append(cont)
                
            # dist area smaller or equal to single point or grid size resolution, i.e mode of this dist
            else:
                
                cont = [np.array(modes.numpy()[h], dtype=np.float32)[None, ...]]
                contours.append([np.squeeze(a=cont, axis=0)])
            
        confidence_contours.append(contours)
        
    return confidence_contours


def build_mesh_grid(samples, mesh_resolution):
    """create 2d plain mesh grid
    Args:
        samples (torch.tensor): batch of samples
        mesh_resolution (float): resolution of mesh grid
    Returns:
        torch.tensor: mesh grid
    """
    
    # find size for mesh grid
    limits = [round_min(samples[...,0].min().item()), 
            round_max(samples[...,0].max().item()),
            round_min(samples[...,1].min().item()),
            round_max(samples[...,1].max().item())]
    
    # find max to define range
    range = math.ceil(max(abs(np.array(limits)))+1)
    steps = int(((range + range) / mesh_resolution) + 1)
    
    # build grid
    steps = int(((range + range) / mesh_resolution) + 1)
    xs = torch.linspace(-range, range, steps=steps)
    ys = torch.linspace(-range, range, steps=steps)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([x, y],dim=-1)
    grid = grid.reshape((-1,2))[:,None,:]
    
    return grid, steps, limits, range


def build_confidence_set(gt, gmm, n_sample_points):
    """create confidence maps for batched distributions
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features] or mesh grid [n_grid_points, 1, 2]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        n_sample_points (int): number of samples to get from each distribution
    Returns:
        torch.tensor: batched confidence maps
    """
    
    gt_log_prob = gmm.log_prob(gt)
    samples = gmm.sample(sample_shape=torch.Size([n_sample_points]))
    samples_log_prob = gmm.log_prob(samples)
    idx_mask = (samples_log_prob > gt_log_prob).float()
    conf = torch.sum(idx_mask, 0)/samples.shape[0]
    return conf


def estimate_sharpness(confidence_map, kappa):
    """get sharpness area
    Args:
        confidence_map (torch.tensor): confidence map for single track [n_grid_points, n_horizons]
        kappa (float): confidence level
    Returns:
        torch.tensor: area [n_horizon]
    """
    
    area = torch.where(confidence_map <= kappa, 1.0, 0.0)
    area = area.mean(dim=0)
    return area
    
    
def round_min(x):
    
    if np.sign(x) == -1: return np.ceil(np.abs(x)) * np.sign(x)
    elif np.sign(x) == 1: return np.floor(np.abs(x)) * np.sign(x)
    else: return np.floor(np.abs(x)) * np.sign(x)
    
    
def round_max(x):
    
    return np.ceil(np.abs(x)) * np.sign(x)

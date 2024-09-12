import torch

from tqdm import tqdm
from termcolor import colored


def calc_ade_fde(gt, gmm, k):
    """calculate best of k min ade/fde scores from distribution samples
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        k (int): number k of samples to get from distributions
    Returns:
        float: min_ade, min_fde scores
    """
    
    # get samples from distributions
    print(colored(f" - Calculating min_ADE and min_FDE (K={k}) scores", 'green'))
    predictions = get_samples(gmm=gmm, n_sample_points=k)
    
    # reshape from [k, batch_size, n_horizons, n_features] to [batch_size, n_horizons, k, n_features]
    pr = predictions.permute(1,2,0,3)
    
    # predictions: [batch_size, n_horizons, k, n_features]
    min_ade = compute_ade(predictions=pr, gt=gt)
    min_fde = compute_fde(predictions=pr, gt=gt)
    
    return min_ade, min_fde


def calc_ade_fde_for_cls(gt, gmm, k, n_sample_points, n_forecast_horizons, n_features, confidence_levels):
    """calculate best of k min ade/fde scores from distribution samples at defined confidence levels
    Args:
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        k (int): number k of samples to get from distributions
        n_sample_points (int): number of samples to get from each distribution
        n_forecast_horizons (int): number of discrete forecast horizons
        n_features (int): number of features
        confidence_levels (list): list object containing a number of user defined confidence levels
    Returns:
        dict: dictionary containing min_ade, min_fde scores at each defined confidence level
    """
    
    print(colored(f" - Calculating min_ADE and min_FDE (K={k}) scores for confidence levels: {confidence_levels}", 'green'))
    n_tracks = gt.shape[0]
    res_dict = {}
    
    # do for each confidence level
    for level in tqdm(confidence_levels):
        
        # get k samples from within defined confidence level of distributions
        k_samples = get_samples_from_confidence_level(gmm=gmm, 
                                                        level=level, 
                                                        k=k, 
                                                        n_sample_points=n_sample_points, 
                                                        n_tracks=n_tracks, 
                                                        n_forecast_horizons=n_forecast_horizons, 
                                                        n_features=n_features)
        
        # predictions: [batch_size, n_horizons, k, n_features]
        # gt: [batch_size, n_horizons, n_features]
        min_ade = compute_ade(predictions=k_samples, gt=gt)
        min_fde = compute_fde(predictions=k_samples, gt=gt)
        
        # save results
        res_dict[f"min_ADE (K={k}) @ CL: {level}:"] = min_ade
        res_dict[f"min_FDE (K={k}) @ CL: {level}:"] = min_fde
    
    return res_dict


def compute_ade(predictions, gt):
    """ batched min ade calculation
    Args:
        predictions (torch.tensor): k predicted samples [batch_size, n_horizons, k, n_features]
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
    Returns:
        float: min ade result
    """
    
    error = torch.linalg.norm(predictions - gt.unsqueeze(2), axis=-1)
    ade = torch.mean(error, axis=1)
    min_ade = torch.min(ade, dim=1).values.mean().item()
    return min_ade


def compute_fde(predictions, gt):
    """ batched min fde calculation
    Args:
        predictions (torch.tensor): k predicted samples [batch_size, n_horizons, k, n_features]
        gt (torch.tensor): batch of ground truth data [batch_size, n_horizons, n_features]
    Returns:
        float: min fde result
    """
    
    fde = torch.linalg.norm(predictions - gt.unsqueeze(2), axis=-1)[:,-1,:]
    min_fde = torch.min(fde, dim=1).values.mean().item()
    return min_fde


def get_samples(gmm, n_sample_points):
    """get n samples from batched distributions
    Args:
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        n_sample_points (int): number of samples to get from each distribution
    Returns:
        torch.tensor: batched samples
    """
    
    samples = gmm.sample(sample_shape=torch.Size([n_sample_points]))
    return samples


def get_samples_from_confidence_level(gmm, level, k, n_sample_points, n_tracks, n_forecast_horizons, n_features):
    """get k samples from batched distributions at defined confidence level
    Args:
        gmm (torch.distribution.MixtureSameFamily): batch of gaussian mixture models [batch_size, n_horizons]
        level (float): confidence level
        k (int): number k of samples to get from distributions
        n_sample_points (int): number of samples to get from each distribution
        n_tracks (int): number of trackes / batch size
        n_forecast_horizons (int): number of discrete forecast horizons
        n_features (int): number of features
    Returns:
        torch.tensor: batched k random samples of defined confidence level
    """
    
    # compute log probabilities for each sample
    tau = 1
    samples = gmm.sample(sample_shape=torch.Size([n_sample_points]))
    samples_log_prob = gmm.log_prob(samples)
    probs = torch.exp(samples_log_prob/tau) / torch.sum(torch.exp(samples_log_prob/tau), dim=0)
    
    # filter samples by confidence/denisty level using probabilities
    thres = int(level * n_sample_points)
    _, indices = probs.topk(k=thres, dim=0, largest=True, sorted=True)
    filtered_samples = torch.gather(samples, 0, indices.unsqueeze(-1).expand(-1, -1, n_forecast_horizons, n_features))
    
    # randomly extract k samples for each track and timestep
    random_k_samples = torch.stack([torch.stack([filtered_samples[torch.randperm(thres)[:k], i, j] for j in range(n_forecast_horizons)]) for i in range(n_tracks)])
    
    # random_k_samples: [batch_size, n_horizons, k, n_features]
    return random_k_samples
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from scipy.stats import gaussian_kde


def calc_kde(gt, samples, n_forecasts_horizons):
    """calculate kde-nll score
    Args:
        gt (torch.tensor): ground truth data [n_horizons, n_features]
        samples (torch.tensor): k predicted samples [k, n_horizons, n_features]
        n_forecast_horizons (int): number of discrete forecast horizons
    Returns:
        float: kde-nll score
    """
    
    kde_ll = 0
    kde_ll_f = 0
    n_u_c = 0
    
    # Per forecast horizon step
    for fh in range(n_forecasts_horizons):
            
        temp = samples[:, fh,:]
        n_unique = len(np.unique(temp, axis=0))
        
        if n_unique > 2:
            
            kde = gaussian_kde(samples[:, fh,:].T)
            
            t = np.clip(kde.logpdf(gt[fh]), a_min=-20, a_max=None)[0]
            
            kde_ll += t
            
            if fh == (n_forecasts_horizons - 1):
                
                kde_ll_f += t
        else:
            
            n_u_c += 1
                
    if n_u_c == n_forecasts_horizons: return 0
    else: return -kde_ll / n_forecasts_horizons
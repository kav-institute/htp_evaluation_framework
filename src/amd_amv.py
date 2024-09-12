import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from math import sqrt, exp
from scipy.special import erf


def calc_amd_amv(gt, gmm, n_forecasts_horizons):
    """calculate amd and amv scores, no batched function
    Args:
        gt (np.array): single prediction ground truth data [n_horizons, n_features]
        gmm (torch.distribution.MixtureSameFamily): single prediction gaussian mixture model [1, n_horizons]
        n_forecast_horizons (int): number of discrete forecast horizons
    Returns:
        float: amd and amv scores
    """
    
    total = 0
    m_collect = []
    gmm_cov_all = 0
    
    # shapes:
    # gmm_weights: [n_forecast_horizons, n_components]
    # gmm_means: [n_forecast_horizons, n_components, n_features]
    # gmm_covariances: [n_forecast_horizons, n_components, n_features, n_features]
    gmm_weights = gmm.mixture_distribution.probs.squeeze().detach().numpy()
    gmm_means = gmm.component_distribution.loc.squeeze().detach().numpy()
    gmm_covariances = gmm.component_distribution.covariance_matrix.squeeze().detach().numpy()
    gt = gt.detach().numpy()
    
    # do per forecast horizon
    for fh in range(n_forecasts_horizons):
        
        center = np.sum(np.multiply(gmm_means[fh], gmm_weights[fh][:,np.newaxis]), axis=0)
        gmm_cov = 0
        
        for cnt in range(len(gmm_means[fh])):
            
            gmm_cov += gmm_weights[fh][cnt] * (gmm_means[fh][cnt] - center)[..., None] @ np.transpose((gmm_means[fh][cnt] - center)[..., None])
            
        gmm_cov = np.sum(gmm_weights[fh][..., None, None] * gmm_covariances[fh], axis=0) + gmm_cov
        dist = calc_mahalanobis_distance(center, gt[fh], len(gmm_weights[fh]), gmm_covariances[fh], gmm_means[fh], gmm_weights[fh]) 
        total += dist
        gmm_cov_all += gmm_cov
        m_collect.append(dist)
        
    gmm_cov_all = gmm_cov_all / n_forecasts_horizons
    amd = total / n_forecasts_horizons
    amv = np.abs(np.linalg.eigvals(gmm_cov_all)).max()
    
    return amd, amv


def calc_mahalanobis_distance(x, y, n_clusters, ccov, cmeans, cluster_p):
    """ calculate mahalanobis distance
    Args:
        x (np.array): distribution center position
        y (np.array): ground truth position
        n_clusters (int): number of distribution components
        ccov (np.array): disribution covariance matrixes
        cmeans (np.array): disribution means
        cluster_p (np.array): disribution weights
    Returns:
        float: mahalanobis distance
    """
    
    v = np.array(x - y)
    Gnum = 0
    Gden = 0
    
    # do for each component
    for i in range(0, n_clusters):
        
        ck = np.linalg.pinv(ccov[i])
        u = np.array(cmeans[i] - y)
        val = ck * cluster_p[i]
        b2 = 1 / (v.T @ ck @ v)
        a = b2 * v.T @ ck @ u
        Z = u.T @ ck @ u - b2 * (v.T @ ck @ u)**2
        pxk = sqrt(np.pi * b2 / 2) * exp(-Z / 2) * (erf((1 - a) / sqrt(2 * b2)) - erf(-a / sqrt(2 * b2)))
        Gnum += val * pxk
        Gden += cluster_p[i] * pxk
        
    G = Gnum / Gden
    mdist = sqrt(v.T @ G @ v)
    
    if np.isnan(mdist):
        
        return 0, 0
    
    return mdist
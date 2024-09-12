import sys
import torch
import numpy as np

from termcolor import colored
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily

sys.path.append("/workspace/repos/src")


class BN_Fit():
    
    def __init__(self, n_forecast_horizons=12, n_components=3, n_iters=100, n_parallel=8):
        
        self.n_iters = n_iters
        self.n_components = n_components
        self.n_forecast_horizons = n_forecast_horizons
        self.n_parallel = n_parallel
        return
    
    
    def fit(self, sample_points_array):
        
        # sample_points_array is array of shape [n_tracks, n_sample_points, n_forcast_horizons, coord_dim]
        gmm_list = list(tqdm(Parallel(return_as="generator", n_jobs=self.n_parallel)(delayed(self.gmm_fit)(sample_points=s) for s in sample_points_array), total=len(sample_points_array)))
        torch_gmm = self.to_pytorch(gmm_list=gmm_list)
        return torch_gmm
    
    
    def gmm_fit(self, sample_points):
        
        # split prediction into discrete forcast horizon steps and make a fit for each individual
        # enroll in parallel
        slices = [sample_points[:, i, :] for i in range(self.n_forecast_horizons)]
        gmm_list = Parallel(n_jobs=self.n_forecast_horizons)(delayed(self.bn_algo)(s) for s in slices)
        
        return gmm_list
    
    
    def bn_algo(self, sample_points):
        
        bic = []
        lowest_bic = np.infty
        cv_types = ['full']
        best_gmm = BayesianGaussianMixture()
        
        for cv_type in cv_types:
            
            gmm = BayesianGaussianMixture(n_components=self.n_components, max_iter=self.n_iters, covariance_type=cv_type)
            gmm.fit(sample_points)
            bic.append(self.calc_bic(model=gmm, X=sample_points))
            
            if bic[-1] < lowest_bic:
                
                lowest_bic = bic[-1]
                best_gmm = gmm
                    
        return best_gmm
    
    
    def calc_bic(self, model, X):
        
        n_params = model.n_components * (model.n_features_in_ * (model.n_features_in_ + 1)) // 2 + model.n_components - 1
        n_samples = len(X)
        llh = model.score(X).sum()
        
        return np.log(n_samples) * n_params - 2 * llh
    
    
    def to_pytorch(self, gmm_list):
        
        # extract the parameters from the EM algo GaussianMixture objects
        weights = []
        mu_x = []
        mu_y = []
        covariances = []
        
        # for each full prediction
        for gmm in gmm_list:
            
            we = []
            mx = []
            my = []
            co= []
            
            # for each forcast horizon
            for g in gmm:
                
                we.append(g.weights_)
                mx.append(g.means_[:,0])
                my.append(g.means_[:,1])
                co.append(g.covariances_)
                
            weights.append(we)
            mu_x.append(mx)
            mu_y.append(my)
            covariances.append(co)
        
        # create a torch.distributions MixtureSameFamily object
        alpha = torch.tensor(np.array(weights))
        mu_x = torch.tensor(np.array(mu_x))
        mu_y = torch.tensor(np.array(mu_y))
        normal = MultivariateNormal(torch.stack([mu_x, mu_y], dim=-1), covariance_matrix=torch.tensor(np.array(covariances)))
        mixture = Categorical(probs=alpha)
        torch_gmm = MixtureSameFamily(mixture, normal)
        
        return torch_gmm
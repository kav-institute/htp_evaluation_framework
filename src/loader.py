import os
import sys
import argparse
import torch
import math
import h5py
import copy
import numpy as np

from em_fit import EM_Fit
from bn_fit import BN_Fit

from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from termcolor import colored

sys.path.append("/workspace/repos/src")


class Numpy_Loader():
    """Numpy data loader class
    """
    
    def __init__(self, input_path, gt_path, k_samples_path, full_samples_path, gmms_path):
        """Initialize numpy loader
        Args:
            input_path (string): input trajectories
            gt_path (string): ground truth trajectories
            k_samples_path (string):  k trajectories for best of k evaluation (only for detemernistic evaluations)
            full_samples_path (string): 1000 trajectory samples (for probabilistic evaluations)
            gmms_path (string): fitted guassian mixture parameters
        """
        
        print(colored(f"Numpy dataloader:", 'white'))
        self.input_path = input_path
        self.k_samples_path = k_samples_path
        self.full_samples_path = full_samples_path
        self.gmms_path = gmms_path
        self.gt_path = gt_path
        
        return
    
    
    def load_input(self):
        """load input trajectories
        Returns:
            torch.tensor: input trajectories [n_tracks, n_input_horizons, n_features]
        """
        
        print(colored(f" - Loading input data from file: {self.input_path}", 'white'))
        return torch.tensor(np.load(self.input_path, allow_pickle=False), dtype=torch.float32)
    
    
    def load_gt(self):
        """load gt trajectories
        Returns:
            torch.tensor: gt trajectories [n_tracks, n_forecasts_horizons, n_features]
        """
        
        print(colored(f" - Loading GT data from file: {self.gt_path}", 'white'))
        return torch.tensor(np.load(self.gt_path, allow_pickle=False), dtype=torch.float32)
    
    
    def write_gt(self, gt_data, path):
        """save gt data to numpy file
        Args:
            gt_data (torch.tensor): gt data
            path (string): destination path to save file
        """
        
        np.save(path, gt_data.detach().numpy())
        return
    
    
    def load_gmms(self):
        """load fitted mixtures
        Returns:
            torch.distributions.MixtureSameFamily: fitted mixtures batch [n_tracks, n_forecasts_horizons]
        """
        
        print(colored(f" - Loading predicition gmm distributions from file: {self.gmms_path}", 'green'))
        
        # load distribution parameters and convert to batched pytorch MixtureSameFamily object
        gmms = np.load(self.gmms_path)
        
        weights = gmms['mixture_weights'].astype(np.float32)
        comp_means = gmms['component_means'].astype(np.float32)
        covariances = gmms['component_stds'].astype(np.float32)
        
        normal = MultivariateNormal(torch.tensor(comp_means), covariance_matrix=torch.tensor(covariances))
        mixture = Categorical(probs=torch.tensor(weights))
        torch_gmms = MixtureSameFamily(mixture, normal)
            
        print(colored(f" - load complete", 'green'))
        return torch_gmms
    
    
    def load_samples(self):
        """load full samples
        Returns:
            torch.tensor: batched samples
        """
        
        print(colored(f" - Loading prediction sample points from file: {self.full_samples_path}", 'green')) 
        
        # load sample points
        samples = torch.tensor(np.load(self.full_samples_path, allow_pickle=True), dtype=torch.float32)
        
        print(colored(f" - load complete", 'green'))
        return samples
    
    
    def fit_gmms_to_samples(self, method, n_components, n_forecast_horizons, samples, n_iters, n_parallel):
        """apply gaussian fitting to full samples to get mixture models
        Args:
            method (string): fitting method
            n_components (int): number of distribution components
            n_forecast_horizons (int): number of discrete forecast horizons
            samples (torch.tensor): full samples
            n_iters (int): number of fit iterations
            n_parallel (int): number of parallel multiprocessing processes
        Returns:
            torch.distributions.MixtureSameFamily: fitted mixtures batch [n_tracks, n_forecasts_horizons]
        """
        
        print(colored(f" - Applying: {method} algo to create distribution data from sample points", 'green')) 
        
        if method == 'em': 
            
            algo = EM_Fit(n_forecast_horizons=n_forecast_horizons, n_components=n_components, n_iters=n_iters, n_parallel=n_parallel)
            torch_gmms = algo.fit(sample_points_array=samples)
            
        elif method == 'bn': 
            
            algo = BN_Fit(n_forecast_horizons=n_forecast_horizons, n_components=n_components, n_iters=n_iters, n_parallel=n_parallel)
            torch_gmms = algo.fit(sample_points_array=samples)
        
        print(colored(f" - fit complete", 'green'))
        return torch_gmms
    
    
    def write_gmms(self, gmms, path):
        """save fitted mixtures to file
        Args:
            gmms (torch.distributions.MixtureSameFamily): fitted mixtures batch [n_tracks, n_forecasts_horizons]
            path (string): destination path to save data
        """
        
        print(colored(f" - saving prediction gmm data to: {path}", 'green'))
        
        weights = []
        comp_means = []
        covariances = []
        
        for idx in range(gmms.batch_shape[0]):
            
            weights.append(gmms.mixture_distribution.probs[idx].detach().numpy().astype(np.float32))
            comp_means.append(gmms.component_distribution.loc[idx].detach().numpy().astype(np.float32))
            covariances.append(gmms.component_distribution.covariance_matrix[idx].detach().numpy().astype(np.float32))
            
        # save all parameters to .npz file
        np.savez(path, mixture_weights=weights, component_means=comp_means, component_stds=covariances)
        
        print(colored(f" - save complete", 'green'))
        return
    
    
class Pth_Loader():
    
    def __init__(self, input_path, gt_path, k_samples_path, full_samples_path, gmms_path):
        """Initialize pth loader
        Args:
            input_path (string): input trajectories
            gt_path (string): ground truth trajectories
            k_samples_path (string):  k trajectories for best of k evaluation (only for detemernistic evaluations)
            full_samples_path (string): 1000 trajectory samples (for probabilistic evaluations)
            gmms_path (string): fitted guassian mixture parameters
        """
        
        print(colored(f"Pth dataloader:", 'white'))
        self.input_path = input_path
        self.k_samples_path = k_samples_path
        self.full_samples_path = full_samples_path
        self.gmms_path = gmms_path
        self.gt_path = gt_path
        
        return
    
    
    def load_input(self):
        """load input trajectories
        Returns:
            torch.tensor: input trajectories [n_tracks, n_input_horizons, n_features]
        """
        
        print(colored(f" - Loading input data from file: {self.input_path}", 'white'))
        return torch.load(self.input_path)
    
    
    def load_gt(self):
        """load gt trajectories
        Returns:
            torch.tensor: gt trajectories [n_tracks, n_forecasts_horizons, n_features]
        """
        
        print(colored(f" - Loading GT data from file: {self.gt_path}", 'white'))
        return torch.load(self.gt_path)
    
    
    def write_gt(self, gt_data, path):
        """save gt data to numpy file
        Args:
            gt_data (torch.tensor): gt data
            path (string): destination path to save file
        """
        
        torch.save(gt_data, path)
        return
    
    
    def load_gmms(self):
        """load fitted mixtures
        Returns:
            torch.distributions.MixtureSameFamily: fitted mixtures batch [n_tracks, n_forecasts_horizons]
        """
        
        print(colored(f" - Loading predicition distributions from file: {self.gmms_path}", 'green'))
        
        # load distribution parameters
        torch_gmm = torch.load(self.gmms_path)
            
        print(colored(f" - load complete", 'green'))
        return torch_gmm
    
    
    def load_samples(self):
        """load full samples
        Returns:
            torch.tensor: batched samples
        """
        
        print(colored(f" - Loading prediction sample points from file: {self.full_samples_path}", 'green')) 
        
        # load sample points
        samples = torch.load(self.full_samples_path)
        
        print(colored(f" - load complete", 'green'))
        return samples
    
    
    def fit_gmms_to_samples(self, method, n_components, n_forecast_horizons, samples, n_iters, n_parallel):
        """apply gaussian fitting to full samples to get mixture models
        Args:
            method (string): fitting method
            n_components (int): number of distribution components
            n_forecast_horizons (int): number of discrete forecast horizons
            samples (torch.tensor): full samples
            n_iters (int): number of fit iterations
            n_parallel (int): number of parallel multiprocessing processes
        Returns:
            torch.distributions.MixtureSameFamily: fitted mixtures batch [n_tracks, n_forecasts_horizons]
        """
        
        print(colored(f" - Applying: {method} algo to create distribution data from sample points", 'green')) 
        
        if method == 'em': 
            
            algo = EM_Fit(n_forecast_horizons=n_forecast_horizons, n_components=n_components, n_iters=n_iters, n_parallel=n_parallel)
            torch_gmms = algo.fit(sample_points_array=samples)
            
        elif method == 'bn': 
            
            algo = BN_Fit(n_forecast_horizons=n_forecast_horizons, n_components=n_components, n_iters=n_iters, n_parallel=n_parallel)
            torch_gmms = algo.fit(sample_points_array=samples)
        
        print(colored(f" - fit complete", 'green'))
        return torch_gmms
    
    
    def write_gmms(self, gmms, path):
        """save fitted mixtures to file
        Args:
            gmms (torch.distributions.MixtureSameFamily): fitted mixtures batch [n_tracks, n_forecasts_horizons]
            path (string): destination path to save data
        """
        
        print(colored(f" - saving prediction data to: {path}", 'green'))
        
        torch.save(gmms, path)
        
        print(colored(f" - save complete", 'green'))
        return
    
    
class H5_Loader():
    
    def __init__(self, input_path, gt_path, k_samples_path, full_samples_path, gmms_path):
        """Initialize pth loader
        Args:
            input_path (string): input trajectories
            gt_path (string): ground truth trajectories
            k_samples_path (string):  k trajectories for best of k evaluation (only for detemernistic evaluations)
            full_samples_path (string): 1000 trajectory samples (for probabilistic evaluations)
            gmms_path (string): fitted guassian mixture parameters
        """
        
        print(colored(f"Pth dataloader:", 'white'))
        self.input_path = input_path
        self.k_samples_path = k_samples_path
        self.full_samples_path = full_samples_path
        self.gmms_path = gmms_path
        self.gt_path = gt_path
        
        return
    
    
    def load_input(self):
        """load input trajectories
        Returns:
            torch.tensor: input trajectories [n_tracks, n_input_horizons, n_features]
        """
        
        # h5 input gt data is called "gt/data"
        with h5py.File(self.input_path) as f: 
            d = np.array(f['input/data'])
            
        f.close()
        return torch.tensor(d)
    
    
    def load_gt(self):
        """load gt trajectories
        Returns:
            torch.tensor: gt trajectories [n_tracks, n_forecasts_horizons, n_features]
        """
        
        print(colored(f" - Loading GT data from file: {self.gt_path}", 'white'))
        
        # h5 input gt data is called "gt/data"
        with h5py.File(self.gt_path) as f: 
            d = np.array(f['gt/data'])
            
        f.close()
        return torch.tensor(d)
    
    
    def write_gt(self, gt_data, path):
        """save gt data to numpy file
        Args:
            gt_data (torch.tensor): gt data
            path (string): destination path to save file
        """
        
        # h5 input gt data is called "gt/data"
        with h5py.File(path, 'w') as f: 
            f.create_dataset('gt/data', data=gt_data.numpy())
            
        return
    
    
    def load_gmms(self):
        """load fitted mixtures
        Returns:
            torch.distributions.MixtureSameFamily: fitted mixtures batch [n_tracks, n_forecasts_horizons]
        """
        
        print(colored(f" - Loading predicition distributions from file: {self.gmms_path}", 'green'))
        
        # Load the gmm objects from h5 file
        # h5 input predictions as didts is called "gmm/weights, gmm/means, and gmm/covariances"
        with h5py.File(self.gmms_path, 'r') as f:
            weights = f['gmm/weights'][:]
            comp_means = f['gmm/means'][:]
            covariances = f['gmm/covariances'][:]
            
        f.close()
        
        normal = MultivariateNormal(torch.tensor(comp_means), covariance_matrix=torch.tensor(covariances))
        mixture = Categorical(probs=torch.tensor(weights))
        torch_gmm = MixtureSameFamily(mixture, normal)
            
        print(colored(f" - load complete", 'green'))
        return torch_gmm
    
    
    def load_samples(self):
        """load full samples
        Returns:
            torch.tensor: batched samples
        """
        
        print(colored(f" - Loading prediction sample points from file: {self.full_samples_path}", 'green')) 
        
        # h5 input gt data is called "gt/data"
        with h5py.File(self.full_samples_path) as f: 
            d = np.array(f['pred/data'])
            
        f.close()
        return torch.tensor(d)
    
    
    def fit_gmms_to_samples(self, method, n_components, n_forecast_horizons, samples, n_iters, n_parallel):
        """apply gaussian fitting to full samples to get mixture models
        Args:
            method (string): fitting method
            n_components (int): number of distribution components
            n_forecast_horizons (int): number of discrete forecast horizons
            samples (torch.tensor): full samples
            n_iters (int): number of fit iterations
            n_parallel (int): number of parallel multiprocessing processes
        Returns:
            torch.distributions.MixtureSameFamily: fitted mixtures batch [n_tracks, n_forecasts_horizons]
        """
        
        print(colored(f" - Applying: {method} algo to create distribution data from sample points", 'green')) 
        
        if method == 'em': 
            
            algo = EM_Fit(n_forecast_horizons=n_forecast_horizons, n_components=n_components, n_iters=n_iters, n_parallel=n_parallel)
            torch_gmms = algo.fit(sample_points_array=samples)
            
        elif method == 'bn': 
            
            algo = BN_Fit(n_forecast_horizons=n_forecast_horizons, n_components=n_components, n_iters=n_iters, n_parallel=n_parallel)
            torch_gmms = algo.fit(sample_points_array=samples)
        
        print(colored(f" - fit complete", 'green'))
        return torch_gmms
    
    
    def write_gmms(self, gmms, path):
        """save fitted mixtures to file
        Args:
            gmms (torch.distributions.MixtureSameFamily): fitted mixtures batch [n_tracks, n_forecasts_horizons]
            path (string): destination path to save data
        """
        
        print(colored(f" - saving prediction data to: {path}", 'green'))
        
        weights = []
        comp_means = []
        covariances = []
        
        for idx in range(gmms.batch_shape[0]):
            
            weights.append(gmms.mixture_distribution.probs[idx].detach().numpy())
            comp_means.append(gmms.component_distribution.loc[idx].detach().numpy())
            covariances.append(gmms.component_distribution.covariance_matrix[idx].detach().numpy())
            
        with h5py.File(path, 'w') as f: 
            f.create_dataset('gmm/weights', data=weights)
            f.create_dataset('gmm/means', data=comp_means)
            f.create_dataset('gmm/covariances', data=covariances)
            
        f.close()
        
        print(colored(f" - save complete", 'green'))
        return